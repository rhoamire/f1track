import numpy as np
import matplotlib.pyplot as plt
import fastf1
from scipy.signal import savgol_filter

fastf1.Cache.enable_cache('f1_cache')

class Track:
    def __init__(self, track_name, year=2023, session_type='Q'):
        """Load track data from FastF1 and create a track model."""
        self.name = track_name

        # Load track data from FastF1
        session = fastf1.get_session(year, track_name, session_type)
        session.load()

        # Get the fastest lap
        fastest_lap = session.laps.pick_fastest()
        self.telemetry = fastest_lap.get_telemetry()

        # Extract track coordinates
        self.x = self.telemetry['X'].values
        self.y = self.telemetry['Y'].values

        # Convert to a more usable format and calculate additional properties
        self._process_track_data()

    def _process_track_data(self):
        """Process track data to calculate centerline, curvature, and track width."""
        # Calculate distance along track (s coordinate in Frenet)
        self.s = np.zeros(len(self.x))
        for i in range(1, len(self.x)):
            dx = self.x[i] - self.x[i-1]
            dy = self.y[i] - self.y[i-1]
            self.s[i] = self.s[i-1] + np.sqrt(dx**2 + dy**2)

        self.track_length = self.s[-1]

        # Calculate track headings
        self.headings = np.zeros(len(self.x))
        for i in range(len(self.x)-1):
            dx = self.x[i+1] - self.x[i]
            dy = self.y[i+1] - self.y[i]
            self.headings[i] = np.arctan2(dy, dx)
        self.headings[-1] = self.headings[-2]  # Last point

        # Calculate curvature (simplified approximation)
        self.curvature = np.zeros(len(self.x))
        for i in range(1, len(self.x)-1):
            # Approximate curvature using three points
            x1, y1 = self.x[i-1], self.y[i-1]
            x2, y2 = self.x[i], self.y[i]
            x3, y3 = self.x[i+1], self.y[i+1]

            # Calculate vectors from center point to others
            dx1, dy1 = x1 - x2, y1 - y2
            dx3, dy3 = x3 - x2, y3 - y2

            # Calculate changes in heading
            h1 = np.arctan2(dy1, dx1)
            h3 = np.arctan2(dy3, dx3)
            dh = (h3 - h1 + np.pi) % (2 * np.pi) - np.pi

            # Calculate distance
            ds = (np.sqrt(dx1**2 + dy1**2) + np.sqrt(dx3**2 + dy3**2)) / 2

            # Curvature = change in heading / distance
            if ds > 0:
                self.curvature[i] = dh / ds

        # Smooth curvature
        self.curvature = savgol_filter(self.curvature, 21, 3)

        # Estimate track width (using a default if not available in telemetry)
        if 'Width' in self.telemetry:
            self.width = self.telemetry['Width'].values
        else:
            # Approximate width based on curvature
            # Straights (low curvature) typically have 15m width
            # Tight corners can narrow to about 10m
            base_width = 15  # meters
            min_width = 10   # meters
            self.width = base_width - np.abs(self.curvature) * 50
            self.width = np.clip(self.width, min_width, base_width)

    def frenet_to_cartesian(self, s, d):
        """Convert from Frenet coordinates (s,d) to Cartesian (x,y)."""
        # Normalize s to track length
        s = s % self.track_length

        # Find the closest point on the centerline
        idx = np.argmin(np.abs(self.s - s))

        # Get perpendicular direction to centerline
        heading = self.headings[idx]
        perp_heading = heading + np.pi/2

        # Return point offset by d from centerline
        return self.x[idx] + d * np.cos(perp_heading), self.y[idx] + d * np.sin(perp_heading)

    def cartesian_to_frenet(self, x, y):
        """Convert from Cartesian (x,y) to Frenet coordinates (s,d)."""
        # Find closest point on centerline
        distances = np.sqrt((self.x - x)**2 + (self.y - y)**2)
        idx = np.argmin(distances)

        # Get s coordinate
        s = self.s[idx]

        # Calculate d (signed lateral distance)
        heading = self.headings[idx]
        dx, dy = x - self.x[idx], y - self.y[idx]
        # Vector from centerline to point
        point_heading = np.arctan2(dy, dx)
        # Calculate signed distance
        d = np.sign((point_heading - heading + np.pi) % (2*np.pi) - np.pi) * distances[idx]

        return s, d

    def get_curvature_at_s(self, s):
        """Get track curvature at specified s coordinate."""
        s = s % self.track_length
        idx = np.argmin(np.abs(self.s - s))
        return self.curvature[idx]

    def get_width_at_s(self, s):
        """Get track width at specified s coordinate."""
        s = s % self.track_length
        idx = np.argmin(np.abs(self.s - s))
        return self.width[idx]

    def plot_track(self, racing_line=None):
        """Plot the track and optionally a racing line."""
        plt.figure(figsize=(12, 8))
        plt.plot(self.x, self.y, 'k-', linewidth=1, label='Track Centerline')

        if racing_line is not None:
            # racing_line should be a list of (s, d) coordinates
            x_line, y_line = [], []
            for s, d in racing_line:
                x, y = self.frenet_to_cartesian(s, d)
                x_line.append(x)
                y_line.append(y)
            plt.plot(x_line, y_line, 'r-', linewidth=2, label='Racing Line')

        plt.axis('equal')
        plt.title(f'Track: {self.name}')
        plt.legend()
        plt.grid(True)
        plt.show()
