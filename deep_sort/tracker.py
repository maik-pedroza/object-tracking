# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track, TrackState
from collections import deque
import math
from . import nn_matching  # for cosine distance helper


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3,
                 lost_ttl: int = 30,
                 gating_alpha: float = 0.5):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        # Tracks currently alive (confirmed or tentative)
        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

        # Lost (deleted) tracks kept for potential resurrection.
        # Each entry is a tuple (track, frames_since_deleted)
        self.lost_tracks: list[tuple[Track, int]] = []
        self._lost_ttl = lost_ttl
        self.gating_alpha = gating_alpha

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        # Predict active tracks
        for track in self.tracks:
            track.predict(self.kf)

        # Predict lost tracks as well so that their state stays up to date
        for idx, (lost_track, age) in enumerate(self.lost_tracks):
            lost_track.predict(self.kf)
            # increment age counter
            self.lost_tracks[idx] = (lost_track, age + 1)
        # Purge lost tracks that exceeded TTL
        self.lost_tracks = [t for t in self.lost_tracks if t[1] <= self._lost_ttl]

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

            # Move newly deleted tracks to lost list (age=0)
            if self.tracks[track_idx].is_deleted():
                self.lost_tracks.append((self.tracks[track_idx], 0))

        # --------------------------------------------------------------
        # Attempt to resurrect lost tracks using remaining detections
        # --------------------------------------------------------------
        resurrect_matches, unmatched_detections = self._reactivate_lost_tracks(
            detections, unmatched_detections)

        # For each resurrected track, perform standard update and add back
        for track_obj, det_idx in resurrect_matches:
            track_obj.state = TrackState.Confirmed
            track_obj.time_since_update = 0
            track_obj.hits += 1
            # ensure track list contains it only once
            if track_obj not in self.tracks:
                self.tracks.append(track_obj)
            track_obj.update(self.kf, detections[det_idx])

        # Initiate new tracks for detections that remain unmatched
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices, alpha=self.gating_alpha)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    # ------------------------------------------------------------------
    # Lost track reactivation helpers
    # ------------------------------------------------------------------
    def _reactivate_lost_tracks(self, detections, unmatched_detection_indices):
        """Attempt to match lost tracks with currently unmatched detections
        using appearance features (cosine distance). Returns updated lists.
        """

        if not self.lost_tracks or not unmatched_detection_indices:
            return [], unmatched_detection_indices

        # Prepare features and cost matrix
        detection_features = np.array([
            detections[i].feature for i in unmatched_detection_indices
        ])

        lost_track_objs = [t for (t, age) in self.lost_tracks]

        # Build cost matrix based on cosine distance
        cost_matrix = np.zeros((len(lost_track_objs), len(detection_features)), dtype=np.float32)
        for row, trk in enumerate(lost_track_objs):
            if trk.smoothed_feature is None:
                # Fallback to latest feature in buffer if EMA not ready
                trk_feat = trk.embedding_buffer[-1]
            else:
                trk_feat = trk.smoothed_feature
            # use helper cosine distance
            dist = nn_matching._cosine_distance(
                trk_feat[None, :], detection_features, data_is_normalized=False
            )[0]
            cost_matrix[row, :] = dist

        # Gating by distance threshold
        cost_matrix[cost_matrix > self.metric.matching_threshold] = self.metric.matching_threshold + 1e-5

        # Hungarian assignment
        indices = np.asarray(linear_assignment.linear_sum_assignment(cost_matrix)).T

        matches, resurrected_tracks = [], []
        unmatched_dets = list(unmatched_detection_indices)

        for row, col in indices:
            if cost_matrix[row, col] > self.metric.matching_threshold:
                continue
            track = lost_track_objs[row]
            det_idx = unmatched_detection_indices[col]
            matches.append((track, det_idx))
            resurrected_tracks.append(track)
            if det_idx in unmatched_dets:
                unmatched_dets.remove(det_idx)

        # Remove resurrected tracks from lost list
        self.lost_tracks = [(t, age) for (t, age) in self.lost_tracks if t not in resurrected_tracks]

        return matches, unmatched_dets

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
