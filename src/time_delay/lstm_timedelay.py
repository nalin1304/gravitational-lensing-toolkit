"""
LSTM-Based Time Delay Measurement

Implements the HOLISMOKES XII method using LSTM-FCNN for improved
time delay measurements in gravitational lensing.

Reference:
---------
Huber, S. & Suyu, S.H. (2024), "HOLISMOKES XII: Time-delay measurements of
supernovae with LSTM neural networks", A&A, 688, A64
arXiv:2403.08029

The LSTM-FCNN architecture achieves ~0.7 day precision, 3× better than
Random Forest methods.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from scipy.signal import correlate
from scipy.interpolate import interp1d


class LSTMTimeDelayEstimator(nn.Module):
    """
    LSTM-FCNN architecture for time delay estimation.

    Combines Long Short-Term Memory (LSTM) layers for temporal feature
    extraction with Fully Connected (FC) layers for regression.

    Architecture:
    - Input: Light curves (batch, seq_len, n_bands)
    - LSTM layers: Extract temporal features
    - FC layers: Map to time delay prediction
    - Output: Time delay and uncertainty

    Reference
    ---------
    Huber & Suyu (2024), A&A, 688, A64
    """

    def __init__(
        self,
        input_size: int = 4,  # Number of photometric bands
        hidden_size: int = 128,
        num_lstm_layers: int = 2,
        fc_hidden_size: int = 256,
        dropout: float = 0.2,
    ):
        super(LSTMTimeDelayEstimator, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM layers for temporal feature extraction
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True,
        )

        # FC layers for time delay regression
        # Bidirectional LSTM outputs 2*hidden_size
        lstm_output_size = 2 * hidden_size

        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, fc_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size // 2, 2),  # [time_delay, uncertainty]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Parameters
        ----------
        x : torch.Tensor
            Light curves, shape (batch, seq_len, n_bands)

        Returns
        -------
        output : torch.Tensor
            Time delay and uncertainty, shape (batch, 2)
        """
        # LSTM feature extraction
        lstm_out, _ = self.lstm(x)

        # Use final hidden state
        features = lstm_out[:, -1, :]

        # FC regression
        output = self.fc_layers(features)

        return output


class TimeDelayMeasurement:
    """
    Time delay measurement using LSTM or cross-correlation.

    Combines traditional cross-correlation with modern LSTM
    deep learning for robust time delay estimation.
    """

    def __init__(self, method: str = "lstm", model_path: Optional[str] = None):
        """
        Initialize time delay measurement.

        Parameters
        ----------
        method : str
            'lstm' or 'correlation'
        model_path : str, optional
            Path to pre-trained LSTM model
        """
        self.method = method
        self.model = None

        if method == "lstm":
            self.model = LSTMTimeDelayEstimator()
            if model_path:
                self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

    def measure(
        self,
        lightcurve_A: np.ndarray,
        lightcurve_B: np.ndarray,
        times: np.ndarray,
        uncertainties_A: Optional[np.ndarray] = None,
        uncertainties_B: Optional[np.ndarray] = None,
        use_ml: bool = True,
    ) -> Dict:
        """
        Measure time delay between two light curves.

        Parameters
        ----------
        lightcurve_A : np.ndarray
            Light curve of image A (magnitudes or flux)
        lightcurve_B : np.ndarray
            Light curve of image B
        times : np.ndarray
            Observation times (days)
        uncertainties_A : np.ndarray, optional
            Measurement uncertainties for A
        uncertainties_B : np.ndarray, optional
            Measurement uncertainties for B
        use_ml : bool
            Use LSTM if available, otherwise use correlation

        Returns
        -------
        result : dict
            Dictionary containing:
            - 'time_delay': Measured time delay in days
            - 'uncertainty': 1σ uncertainty
            - 'method': Method used
            - 'goodness_of_fit': Quality metric

        Reference
        ---------
        Huber & Suyu (2024), A&A, 688, A64
        """
        if use_ml and self.model is not None:
            return self._measure_lstm(
                lightcurve_A, lightcurve_B, times, uncertainties_A, uncertainties_B
            )
        else:
            return self._measure_correlation(
                lightcurve_A, lightcurve_B, times, uncertainties_A, uncertainties_B
            )

    def _measure_lstm(
        self,
        lc_A: np.ndarray,
        lc_B: np.ndarray,
        times: np.ndarray,
        err_A: Optional[np.ndarray],
        err_B: Optional[np.ndarray],
    ) -> Dict:
        """
        Measure using LSTM-FCNN.

        Implements the HOLISMOKES XII method.
        """
        # Prepare input features
        # Combine light curves and uncertainties into feature vector
        n_points = len(times)

        # Normalize light curves
        lc_A_norm = (lc_A - np.mean(lc_A)) / (np.std(lc_A) + 1e-10)
        lc_B_norm = (lc_B - np.mean(lc_B)) / (np.std(lc_B) + 1e-10)

        # Create feature tensor
        if err_A is not None and err_B is not None:
            features = np.stack(
                [
                    lc_A_norm,
                    lc_B_norm,
                    err_A / (np.std(lc_A) + 1e-10),
                    err_B / (np.std(lc_B) + 1e-10),
                ],
                axis=-1,
            )
        else:
            # Pad with zeros if no uncertainties
            features = np.stack(
                [lc_A_norm, lc_B_norm, np.zeros(n_points), np.zeros(n_points)], axis=-1
            )

        # Convert to torch tensor
        x = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension

        # Predict
        with torch.no_grad():
            prediction = self.model(x)

        time_delay = prediction[0, 0].item()
        uncertainty = prediction[0, 1].item()

        # Ensure uncertainty is positive
        uncertainty = max(uncertainty, 0.1)  # Minimum 0.1 day uncertainty

        return {
            "time_delay": time_delay,
            "uncertainty": uncertainty,
            "method": "LSTM-FCNN",
            "goodness_of_fit": self._compute_goodness(lc_A, lc_B, time_delay, times),
            "reference": "Huber & Suyu (2024), A&A, 688, A64",
        }

    def _measure_correlation(
        self,
        lc_A: np.ndarray,
        lc_B: np.ndarray,
        times: np.ndarray,
        err_A: Optional[np.ndarray],
        err_B: Optional[np.ndarray],
    ) -> Dict:
        """
        Measure using cross-correlation (traditional method).

        Uses discrete correlation function with interpolation.
        """
        # Interpolate to common grid
        t_common = np.linspace(times.min(), times.max(), 1000)

        interp_A = interp1d(times, lc_A, kind="cubic", fill_value="extrapolate")
        interp_B = interp1d(times, lc_B, kind="cubic", fill_value="extrapolate")

        lc_A_interp = interp_A(t_common)
        lc_B_interp = interp_B(t_common)

        # Cross-correlation
        correlation = correlate(lc_A_interp, lc_B_interp, mode="full")
        lags = np.arange(-len(lc_A_interp) + 1, len(lc_A_interp))

        # Convert lags to time delays
        dt = np.mean(np.diff(t_common))
        time_delays = lags * dt

        # Find peak
        peak_idx = np.argmax(correlation)
        time_delay = time_delays[peak_idx]

        # Estimate uncertainty from peak width
        # Fit Gaussian to peak
        peak_region = correlation[peak_idx - 10 : peak_idx + 11]
        if len(peak_region) > 5:
            # Simple width estimate
            half_max = np.max(correlation) / 2
            above_half = np.abs(correlation - half_max)
            width_idx = np.argmin(above_half)
            uncertainty = abs(time_delays[width_idx] - time_delay)
        else:
            uncertainty = 1.0  # Default 1 day uncertainty

        return {
            "time_delay": time_delay,
            "uncertainty": uncertainty,
            "method": "Cross-correlation",
            "goodness_of_fit": np.max(correlation) / len(t_common),
            "reference": "Traditional method (PyCS, Barry et al. 2020)",
        }

    def _compute_goodness(
        self, lc_A: np.ndarray, lc_B: np.ndarray, time_delay: float, times: np.ndarray
    ) -> float:
        """Compute goodness of fit metric."""
        # Shift lightcurve B by time_delay
        interp_B = interp1d(
            times, lc_B, kind="cubic", bounds_error=False, fill_value=np.nan
        )
        lc_B_shifted = interp_B(times - time_delay)

        # Remove NaN values
        valid = ~np.isnan(lc_B_shifted)
        if np.sum(valid) < 2:
            import warnings
            warnings.warn(
                "Insufficient valid data points for correlation calculation (< 2). "
                "Returning NaN instead of 0.0 to indicate invalid result."
            )
            return np.nan

        # Pearson correlation coefficient
        corr = np.corrcoef(lc_A[valid], lc_B_shifted[valid])[0, 1]

        return corr


# Convenience function
def measure_time_delay(
    lightcurve_A: np.ndarray,
    lightcurve_B: np.ndarray,
    times: np.ndarray,
    method: str = "lstm",
    model_path: Optional[str] = None,
    uncertainties_A: Optional[np.ndarray] = None,
    uncertainties_B: Optional[np.ndarray] = None,
) -> Dict:
    """
    Measure time delay between lensed images.

    Parameters
    ----------
    lightcurve_A : np.ndarray
        Light curve of image A
    lightcurve_B : np.ndarray
        Light curve of image B
    times : np.ndarray
        Observation times (days)
    method : str
        'lstm' or 'correlation'
    model_path : str, optional
        Path to pre-trained model
    uncertainties_A : np.ndarray, optional
        Uncertainties for A
    uncertainties_B : np.ndarray, optional
        Uncertainties for B

    Returns
    -------
    result : dict
        Time delay measurement with uncertainty

    Example
    -------
    >>> times = np.linspace(0, 100, 50)
    >>> lc_A = np.sin(2 * np.pi * times / 20)
    >>> lc_B = np.sin(2 * np.pi * (times - 5) / 20)  # 5-day delay
    >>> result = measure_time_delay(lc_A, lc_B, times)
    >>> print(f"Time delay: {result['time_delay']:.2f} ± {result['uncertainty']:.2f} days")

    Reference
    ---------
    Huber & Suyu (2024), "HOLISMOKES XII: Time-delay measurements of
    supernovae with LSTM neural networks", A&A, 688, A64
    """
    measurer = TimeDelayMeasurement(method=method, model_path=model_path)
    return measurer.measure(
        lightcurve_A, lightcurve_B, times, uncertainties_A, uncertainties_B
    )
