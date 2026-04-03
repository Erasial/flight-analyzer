from app.parsers.base import DataParser
import pandas as pd
from pymavlink import mavutil

class BinaryDataParser(DataParser):
    def parse(self, file_path: str) -> dict[str, pd.DataFrame]:
        mlog = mavutil.mavlink_connection(file_path)
    
        data = {}

        while True:
            msg = mlog.recv_match()
            
            if msg is None:
                break
                
            msg_type = msg.get_type()
            
            if msg_type == 'FMT':
                continue

            msg_dict = msg.to_dict()
            
            if 'mavpackettype' in msg_dict:
                del msg_dict['mavpackettype']

            if msg_type not in data:
                data[msg_type] = []
            
            data[msg_type].append(msg_dict)

        dataframes = {msg_type: pd.DataFrame(msgs) for msg_type, msgs in data.items()}
        
        return dataframes
