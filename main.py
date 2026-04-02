from app.core.utils import wgs84_to_enu
from app.parsers.binary import BinaryDataParser
from app.services.analyzer import AnalysisService
from visualization.flight_plotter import plot_flight_path_3d

if __name__ == "__main__":
    file_path = "data/00000019.BIN"
    
    parser = BinaryDataParser()
    analyzer = AnalysisService()
    
    dataframes = parser.parse(file_path)
    
    df_gps = dataframes.get('GPS')
    df_gps = analyzer.filter_gps_low_quality_samples(df_gps, min_status=3, require_positive_gwk=True)
    df_gps = wgs84_to_enu(df_gps)
    
    df_imu = dataframes.get('IMU')
    df_imu = analyzer.filter_imu_module(df_imu, imu_index=0)
    
    
    print(df_gps.to_string())
    
    flght_duration = analyzer.get_flight_duration(df_gps)
    distance_traveled = analyzer.get_distance_traveled(df_gps)
    
    max_vertical_speed = analyzer.get_max_vertical_speed(df_gps)
    max_horizontal_speed = analyzer.get_max_horizontal_speed(df_gps)
    max_altitude = analyzer.get_max_altitude(df_gps)
    
    max_acceleration = analyzer.get_max_acceleration(df_imu)
    max_acceleration_x = max_acceleration.get('AccX', 'N/A')
    max_acceleration_y = max_acceleration.get('AccY', 'N/A')
    max_acceleration_z = max_acceleration.get('AccZ', 'N/A')
    
    print(f"Flight Duration: {flght_duration:.2f} seconds")
    print(f"Distance Traveled: {distance_traveled:.2f} meters")
    print(f"Max horizontal speed: {max_horizontal_speed:.2f} m/s")
    print(f"Max vertical speed: {max_vertical_speed:.2f} m/s")
    print(f"Max altitude: {max_altitude:.2f} m")
    
    print(f"Max accelerometer X: {max_acceleration_x:.2f} m/s²")
    print(f"Max accelerometer Y: {max_acceleration_y:.2f} m/s²")
    print(f"Max accelerometer Z: {max_acceleration_z:.2f} m/s²")
    
    print(f"Sample rate GPS: {analyzer.get_sample_rate(df_gps):.2f} Hz")
    print(f"Sample rate IMU: {analyzer.get_sample_rate(df_imu):.2f} Hz")
    
    plot_flight_path_3d(df_gps, output_html="flight_trajectory_enu.html", auto_open=False)
