# vim: expandtab:ts=4:sw=4

import numpy as np
from collections import deque


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None,
                 buffer_size: int = 30,
                 ema_alpha: float = 0.1):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative

        # --- Re-identification buffer -----------------------------------
        # Keep a rolling history of appearance embeddings for multi-frame
        # fusion.  ``embedding_buffer`` has fixed length ``buffer_size``.
        # ``smoothed_feature`` keeps an exponential moving average (EMA)
        # that will be used by the distance metric, providing a single
        # fused descriptor per track.
        self.embedding_buffer = deque(maxlen=buffer_size)
        self.smoothed_feature = None  # type: np.ndarray | None

        # Features pushed each frame and later consumed by Tracker to update
        # the global metric.
        self.features = []

        if feature is not None:
            feature = np.asarray(feature, dtype=np.float32)
            self.embedding_buffer.append(feature)
            self.smoothed_feature = feature
            self.features.append(self.smoothed_feature)

        # EMA smoothing hyper-parameter.
        self._ema_alpha = ema_alpha

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        # --------------------------------------------------------------
        # Update appearance buffer and smoothed descriptor
        # --------------------------------------------------------------
        new_feat = np.asarray(detection.feature, dtype=np.float32)
        self.embedding_buffer.append(new_feat)

        if self.smoothed_feature is None:
            self.smoothed_feature = new_feat
        else:
            # Exponential Moving Average fusion
            self.smoothed_feature = (
                (1.0 - self._ema_alpha) * self.smoothed_feature
                + self._ema_alpha * new_feat
            )

        # L2-normalize to keep cosine distances meaningful
        norm = np.linalg.norm(self.smoothed_feature)
        if norm > 0:
            self.smoothed_feature = self.smoothed_feature / norm

        # Use the fused descriptor for metric update
        self.features.append(self.smoothed_feature)

        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
