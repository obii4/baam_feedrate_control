import numpy as np
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


class PITemperatureController:
    """
    PI Controller for 3D printer temperature control via feed rate adjustment.
    
    Uses Internal Model Control (IMC) tuning methodology with anti-windup,
    rate limiting, and deadband features.
    """
    
    def __init__(self, 
                 setpoint: float = 140.0,
                 nominal_feed: float = 100.0,
                 feed_limits: Tuple[float, float] = (50.0, 200.0),
                 max_feed_change: float = 10.0,
                 temp_deadband: float = 5.0,
                 min_layer_time: float = 60.0):
        """
        Initialize PI controller with operational parameters.
        
        Args:
            setpoint: Target temperature in °C
            nominal_feed: Baseline feed rate in %
            feed_limits: (min, max) feed rate limits in %
            max_feed_change: Maximum feed rate change per layer in %
            temp_deadband: Temperature deadband for control action in °C
            min_layer_time: Minimum allowable layer time in seconds
        """
        # Operational parameters
        self.T_sp = setpoint
        self.u_nom = nominal_feed
        self.u_min, self.u_max = feed_limits
        self.du_max = max_feed_change
        self.dead_T = temp_deadband
        self.t_min = min_layer_time
        
        # Controller gains (will be set during calibration)
        self.Kp_T = 0.0
        self.Ki_T = 0.0
        
        # Process model parameters
        self.K_temp = 0.0  # Temperature gain (°C per % feed)
        self.K_time = 0.0  # Time gain (s per % feed)
        self.T0 = 0.0      # Baseline temperature
        self.y_sp = 0.0    # Baseline layer time
        
        # Controller state
        self.I_T = 0.0     # Integral term
        self.u_prev = nominal_feed
        self.calibrated = False
        
        # Data history for analysis
        self.history = {
            'temps': [],
            'times': [],
            'feeds': [],
            'errors': [],
            'integral_terms': []
        }
    
    def determine_tau_exponential(self, 
                                  times: np.ndarray, 
                                  temps: np.ndarray,
                                  step_idx: Optional[int] = None) -> Dict[str, float]:
        """
        Analyze step response data to determine system parameters using exponential fitting.
        
        Args:
            durations: Array of layer durations (time between measurements)
            temps: Array of temperature measurements
            plot: Whether to plot the results
            
        Returns:
            Dictionary containing:
                - tau: Time constant (seconds)
                - theta: Dead time (seconds)
                - y_initial: Initial steady-state value
                - y_final: Final steady-state value
                - r_squared: Goodness of fit
                - cumulative_time: Cumulative time array
                - fitted_response: Fitted temperature response
        """
        
        # Convert durations to cumulative time
        t = np.cumsum(times)
        y = temps.copy()
        
        # Add t=0 point if not present
        if t[0] > 0:
            t = np.insert(t, 0, 0)
            y = np.insert(y, 0, y[0])
        
        # Define exponential model with dead time
        def exponential_model(t_data: np.ndarray, 
                            y_final: float, 
                            y_initial: float, 
                            tau: float, 
                            theta: float) -> np.ndarray:
            """First-order response with dead time"""
            y_pred = np.zeros_like(t_data)
            mask = t_data > theta
            y_pred[mask] = y_initial + (y_final - y_initial) * (1 - np.exp(-(t_data[mask] - theta) / tau))
            y_pred[~mask] = y_initial
            return y_pred
        
        # Initial parameter estimates
        y_initial = y[0]
        y_final = np.mean(y[step_idx:])  # Average of last 5 points
        tau_guess = (t[-1] - t[0]) / 4
        theta_guess = 5  # Initial dead time guess
        
        # Parameter bounds
        bounds = (
            [y_final - 10, y_initial - 10, 1, 0],  # Lower bounds
            [y_final + 10, y_initial + 10, 1000, 200]  # Upper bounds
        )
        
        try:
            # Perform curve fitting
            popt, pcov = curve_fit(
                exponential_model, t, y,
                p0=[y_final, y_initial, tau_guess, theta_guess],
                bounds=bounds,
                maxfev=5000
            )
            
            y_final_fit, y_initial_fit, tau_fit, theta_fit = popt
            
            # Calculate fitted response
            y_pred = exponential_model(t, *popt)
            
            # Calculate R-squared
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate parameter uncertainties
            perr = np.sqrt(np.diag(pcov))
            tau_std, theta_std = perr[2], perr[3]
            
            # Create results dictionary
            results = {
                'tau': tau_fit,
                'theta': theta_fit,
                'y_initial': y_initial_fit,
                'y_final': y_final_fit,
                'r_squared': r_squared,
                'tau_std': tau_std,
                'theta_std': theta_std,
                'cumulative_time': t,
                'fitted_response': y_pred,
                'measured_temps': y,
                'process_gain': y_final_fit - y_initial_fit,
                'time_to_63_percent': tau_fit + theta_fit
            }
            

            
            return results
            
        except Exception as e:
            print(f"Exponential fitting failed: {e}")
            # Return basic estimates as fallback
            return {
                'tau': None,
                'theta': None,
                'y_initial': y_initial,
                'y_final': y_final,
                'r_squared': None,
                'error': str(e)
            }
            
    
    
    def set_calibration_values(self,
                               K_temp: float,
                               K_time: float,
                               Kp_T: float,
                               Ki_T: float,
                               T0: float,
                               y_sp: float,
                               u_initial: Optional[float] = None) -> Dict[str, float]:
        """
        Directly set calibration values without running calibration process.
        
        Args:
            K_temp: Temperature gain (°C per % feed)
            K_time: Time gain (s per % feed)
            Kp_T: Proportional gain
            Ki_T: Integral gain
            T0: Baseline temperature
            y_sp: Baseline layer time
            u_initial: Initial feed rate (defaults to nominal)
            
        Returns:
            Dictionary of set calibration values
        """
        self.K_temp = K_temp
        self.K_time = K_time
        self.Kp_T = Kp_T
        self.Ki_T = Ki_T
        self.T0 = T0
        self.y_sp = y_sp
        
        if u_initial is not None:
            self.u_prev = u_initial
        
        self.calibrated = True
        
        return {
            'K_temp': self.K_temp,
            'K_time': self.K_time,
            'Kp_T': self.Kp_T,
            'Ki_T': self.Ki_T,
            'T0': self.T0,
            'y_sp': self.y_sp,
            'u_prev': self.u_prev
        }
    
    def calibrate_from_step_test(self, 
                                 times: np.ndarray, 
                                 temps: np.ndarray,
                                 step_magnitude: float = 40.0,
                                 imc_lambda: Optional[float] = None,
                                 tau: Optional[float] = None,
                                 theta: Optional[float] = None,
                                 auto_tune: bool = False,
                                 use_predefined: bool = False,
                                 predefined_values: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calibrate controller from step test data or use predefined values.
        
        Args:
            times: Array of layer times
            temps: Array of previous layer temperatures
            step_magnitude: Feed rate step change magnitude in %
            imc_lambda: IMC tuning parameter (auto-calculated if None)
            tau: Process time constant (auto-calculated if None or auto_tune=True)
            theta: Process dead time (auto-calculated if None or auto_tune=True)
            auto_tune: If True, automatically determine tau and theta from data
            use_predefined: If True, use values from predefined_values dict
            predefined_values: Dict with keys: K_temp, K_time, Kp_T, Ki_T, T0, y_sp, u_initial
            
        Returns:
            Dictionary containing calibration parameters
        """
        # Option 1: Use predefined values
        if use_predefined:
            if predefined_values is None:
                raise ValueError("predefined_values must be provided when use_predefined=True")
            
            required_keys = ['K_temp', 'K_time', 'Kp_T', 'Ki_T', 'T0', 'y_sp']
            missing_keys = [key for key in required_keys if key not in predefined_values]
            
            if missing_keys:
                raise ValueError(f"Missing required keys in predefined_values: {missing_keys}")
            
            return self.set_calibration_values(
                K_temp=predefined_values['K_temp'],
                K_time=predefined_values['K_time'],
                Kp_T=predefined_values['Kp_T'],
                Ki_T=predefined_values['Ki_T'],
                T0=predefined_values['T0'],
                y_sp=predefined_values['y_sp'],
                u_initial=predefined_values.get('u_initial')
            )
        
        # Option 2: Calculate from step test data
        # Detect step change
        diffs = np.diff(times)
        threshold = (times.max() - times.min()) * 0.4
        step_indices = np.where(np.abs(diffs) > threshold)[0]
        
        if len(step_indices) == 0:
            raise ValueError("No step change detected in data")
        
        step_idx = step_indices[0] + 1
        
        # Calculate baselines (before step)
        self.y_sp = times[:step_idx].mean()
        self.T0 = temps[:step_idx].mean()
        
        # Calculate steady-state values (after step)
        y1 = times[step_idx:].mean()
        T1 = temps[step_idx:].mean()
        
        # Calculate process gains
        self.K_time = (y1 - self.y_sp) / step_magnitude
        self.K_temp = (T1 - self.T0) / step_magnitude
        
        # Auto-tune if requested or parameters not provided
        if auto_tune or tau is None or theta is None:
            print("Auto-tuning: Determining tau and theta from step response...")
            tune_results = self.determine_tau_exponential(times, temps, step_idx)
        
            
            if tau is None:
                tau = tune_results['tau']
                
            # chris remember to look at this, currently, theta = imc_lambda
            if theta is None:
                theta = 5
                
        
        # Auto-calculate lambda if not provided
        if imc_lambda is None:
            # Conservative tuning: lambda between theta and 2*theta
            imc_lambda = 5
        
        # IMC-based PI tuning
        self.Kp_T = tau / (abs(self.K_temp) * (imc_lambda + theta))
        self.Ki_T = self.Kp_T / imc_lambda  # Note: Ki = Kp/τi, and τi = τ for IMC
        
        # Set initial feed rate
        self.u_prev = self.u_nom + step_magnitude
        
        self.calibrated = True
        
        calibration_results = {
            'K_temp': self.K_temp,
            'K_time': self.K_time,
            'Kp_T': self.Kp_T,
            'Ki_T': self.Ki_T,
            'T0': self.T0,
            'y_sp': self.y_sp,
            'step_idx': step_idx,
            'tau': tau,
            'theta': theta,
            'imc_lambda': imc_lambda
        }
        
        # Add tuning results if auto-tuning was performed
        if auto_tune or tau is None or theta is None:
            calibration_results['tuning_details'] = tune_results
        
        return calibration_results
    
    def compute_next_feedrate(self, 
                            current_temp: float,
                            current_time: Optional[float] = None,
                            use_deadband: bool = True,
                            use_antiwindup: bool = True) -> Tuple[float, Dict[str, float]]:
        """
        Compute the next feed rate based on current temperature.
        
        Args:
            current_temp: Current previous-layer temperature in °C
            current_time: Current layer time in seconds (optional, for logging)
            use_deadband: Whether to apply deadband logic
            use_antiwindup: Whether to apply anti-windup logic
            
        Returns:
            Tuple of (next_feedrate, diagnostics_dict)
        """
        if not self.calibrated:
            raise RuntimeError("Controller must be calibrated before use")
        
        # Calculate error
        e_T = self.T_sp - current_temp
        
        # Raw PI control law
        u_unsat = self.u_nom + self.Kp_T * e_T + self.Ki_T * self.I_T
        
        # Anti-windup: only integrate if output is not saturated
        if use_antiwindup:
            if self.u_min < u_unsat < self.u_max:
                self.I_T += e_T
        else:
            self.I_T += e_T
        
        # Deadband logic
        if use_deadband and abs(e_T) <= self.dead_T and len(self.history['feeds']) > 0:
            u_target = self.u_prev
        else:
            u_target = u_unsat
        
        # Rate limiting
        du = np.clip(u_target - self.u_prev, -self.du_max, self.du_max)
        u_next = np.clip(self.u_prev + du, self.u_min, self.u_max)
        
        # Update state
        self.u_prev = u_next
        
        # Store history
        self.history['temps'].append(current_temp)
        if current_time is not None:
            self.history['times'].append(current_time)
        self.history['feeds'].append(u_next)
        self.history['errors'].append(e_T)
        self.history['integral_terms'].append(self.I_T)
        
        # Diagnostics
        diagnostics = {
            'error': e_T,
            'proportional_term': self.Kp_T * e_T,
            'integral_term': self.Ki_T * self.I_T,
            'u_unsat': u_unsat,
            'u_target': u_target,
            'rate_limited_change': du,
            'in_deadband': abs(e_T) <= self.dead_T,
            'saturated': u_unsat < self.u_min or u_unsat > self.u_max
        }
        
        return u_next, diagnostics
    
    def predict_temperature(self, feedrate: float) -> float:
        """
        Predict temperature based on feedrate using process model.
        
        Args:
            feedrate: Feed rate in %
            
        Returns:
            Predicted temperature in °C
        """
        return self.T0 + self.K_temp * (feedrate - self.u_nom)
    
    def predict_layer_time(self, feedrate: float) -> float:
        """
        Predict layer time based on feedrate using process model.
        
        Args:
            feedrate: Feed rate in %
            
        Returns:
            Predicted layer time in seconds
        """
        return self.y_sp + self.K_time * (feedrate - self.u_nom)
    
    def reset(self, keep_calibration: bool = True):
        """Reset controller state while optionally keeping calibration."""
        self.I_T = 0.0
        self.u_prev = self.u_nom
        self.history = {
            'temps': [],
            'times': [],
            'feeds': [],
            'errors': [],
            'integral_terms': []
        }
        if not keep_calibration:
            self.calibrated = False
            self.Kp_T = 0.0
            self.Ki_T = 0.0
            self.K_temp = 0.0
            self.K_time = 0.0
    
    def plot_step_test_with_controller(self,
                                       times: np.ndarray,
                                       temps: np.ndarray,
                                       step_magnitude: float = 40.0,
                                       control_start_idx: int = 3,
                                       n_future_layers: int = 15,
                                       figsize: Tuple[float, float] = (12, 10)) -> None:
        """
        Plot step test data and show how controller would continue from a given point.
        
        Args:
            times: Array of measured layer times
            temps: Array of measured temperatures
            step_magnitude: Feed rate step change magnitude in %
            control_start_idx: Index after step where controller takes over
            n_future_layers: Number of future layers to simulate
            figsize: Figure size tuple
        """
        if not self.calibrated:
            raise RuntimeError("Controller must be calibrated before plotting")
        
        # Detect step change
        diffs = np.diff(times)
        threshold = (times.max() - times.min()) * 0.4
        step_indices = np.where(np.abs(diffs) > threshold)[0]
        if len(step_indices) == 0:
            raise ValueError("No step change detected in data")
        step_idx = step_indices[0] + 1
        
        # Calculate measured feed history
        feed_measured = np.ones_like(temps) * self.u_nom
        feed_measured[step_idx:] = self.u_nom + step_magnitude
        
        # Split data into "measured" and "controller takeover" regions
        takeover_idx = step_idx + control_start_idx
        
        # Initialize controller simulation
        self.reset(keep_calibration=True)
        self.u_prev = feed_measured[takeover_idx - 1]
        self.I_T = 0.0  # Reset integral term
        
        # Simulate controller from takeover point
        sim_temps = [temps[takeover_idx - 1]]  # Start with last measured temp
        sim_times = []
        sim_feeds = []
        
        current_temp = temps[takeover_idx - 1]
        
        # Simulate remaining measured points + future layers
        n_total_sim = len(temps) - takeover_idx + n_future_layers
        
        for i in range(n_total_sim):
            # Get controller output
            next_feed, diagnostics = self.compute_next_feedrate(current_temp)
            
            # Predict next temperature
            next_temp = self.predict_temperature(next_feed) + np.random.normal(0, 2.0)
            next_time = self.predict_layer_time(next_feed)
            
            sim_feeds.append(next_feed)
            sim_temps.append(next_temp)
            sim_times.append(next_time)
            
            current_temp = next_temp
        
        # Create layer indices
        layers_measured = np.arange(1, len(temps) + 1)
        layers_controlled = np.arange(takeover_idx, takeover_idx + len(sim_temps))
        
        # Plotting
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Temperature
        ax1 = axes[0]
        # Measured data
        ax1.plot(layers_measured[:takeover_idx], temps[:takeover_idx], 
                'bo-', linewidth=2, markersize=8, label='Measured (before control)')
        ax1.plot(layers_measured[takeover_idx-1:], temps[takeover_idx-1:], 
                'bo--', linewidth=1, markersize=6, alpha=0.5, label='Measured (would continue)')
        
        # Controller simulation
        ax1.plot(layers_controlled, sim_temps, 
                'rs-', linewidth=2, markersize=6, label='Controller prediction')
        
        # Connect the lines
        ax1.plot([layers_measured[takeover_idx-1], layers_controlled[0]], 
                [temps[takeover_idx-1], sim_temps[0]], 
                'k-', linewidth=2, alpha=0.5)
        
        # Reference lines
        ax1.axhline(self.T_sp, color='red', linestyle='--', alpha=0.7, label=f'Setpoint ({self.T_sp}°C)')
        ax1.axvline(step_idx, color='gray', linestyle=':', alpha=0.9, label='Step change')
        ax1.axvline(takeover_idx, color='green', linestyle=':', alpha=0.9, label='Controller starts')
        
        ax1.set_ylabel('Temperature (°C)', fontsize=12)
        ax1.set_title('Step Test vs Controller Prediction', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # Plot 2: Layer Time
        ax2 = axes[1]
        # Measured data
        ax2.plot(layers_measured[:takeover_idx], times[:takeover_idx], 
                'bo-', linewidth=2, markersize=8)
        ax2.plot(layers_measured[takeover_idx-1:], times[takeover_idx-1:], 
                'bo--', linewidth=1, markersize=6, alpha=0.5)
        
        # Controller simulation
        ax2.plot(layers_controlled[1:], sim_times, 
                'rs-', linewidth=2, markersize=6)
        
        # Reference lines
        ax2.axhline(self.y_sp, color='orange', linestyle='--', alpha=0.7, label=f'Baseline ({self.y_sp:.1f}s)')
        ax2.axhline(self.t_min, color='magenta', linestyle=':', alpha=0.7, label=f'Min time ({self.t_min}s)')
        ax2.axvline(step_idx, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(takeover_idx, color='green', linestyle=':', alpha=0.5)
        
        ax2.set_ylabel('Layer Time (s)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        
        # Plot 3: Feed Rate
        ax3 = axes[2]
        # Measured feed
        ax3.plot(layers_measured[:takeover_idx], feed_measured[:takeover_idx], 
                'bo-', linewidth=2, markersize=8, label='Measured feed')
        ax3.plot(layers_measured[takeover_idx-1:], feed_measured[takeover_idx-1:], 
                'bo--', linewidth=1, markersize=6, alpha=0.5, label='Measured (would continue)')
        
        # Controller feed
        ax3.plot(layers_controlled[1:], sim_feeds, 
                'rs-', linewidth=2, markersize=6, label='Controller feed')
        
        # Connect the lines
        if len(sim_feeds) > 0:
            ax3.plot([layers_measured[takeover_idx-1], layers_controlled[1]], 
                    [feed_measured[takeover_idx-1], sim_feeds[0]], 
                    'k-', linewidth=2, alpha=0.5)
        
        # Reference lines
        ax3.axhline(self.u_nom, color='black', linestyle='--', alpha=0.7, label=f'Nominal ({self.u_nom}%)')
        ax3.axhline(self.u_min, color='red', linestyle=':', alpha=0.5, label=f'Limits')
        ax3.axhline(self.u_max, color='red', linestyle=':', alpha=0.5)
        ax3.axvline(step_idx, color='gray', linestyle=':', alpha=0.5)
        ax3.axvline(takeover_idx, color='green', linestyle=':', alpha=0.5)
        
        ax3.set_xlabel('Layer Index', fontsize=12)
        ax3.set_ylabel('Feed Rate (%)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\nController Takeover Analysis:")
        print(f"  Step change at layer: {step_idx}")
        print(f"  Controller starts at layer: {takeover_idx}")
        print(f"  Initial temperature error: {self.T_sp - temps[takeover_idx-1]:.1f}°C")
        if len(sim_temps) > 5:
            final_temp = np.mean(sim_temps[-5:])
            print(f"  Final predicted temperature: {final_temp:.1f}°C")
            print(f"  Final error: {self.T_sp - final_temp:.1f}°C")
