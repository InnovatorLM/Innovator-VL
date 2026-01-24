import socket
import yaml
import os
from pathlib import Path
import pickle
from tqdm import tqdm


config='config.yaml'
with open(config, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

def get_ip_info(ip_file=cfg['hosts']):
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    ips=get_ips(ip_file)
    ip_indx=ips[local_ip]
    ip_num=len(ips)
    return ip_indx,ip_num,local_ip

def get_ips(ip_file):
    with open(ip_file, "r") as f:
        ip_list = [line.strip() for line in f if line.strip()]
        ip_list=sorted(ip_list)
    res={ip:i for i,ip in enumerate(ip_list)}
    return res

def get_split_datas_by_ips(ip_file,datas):
    ip_index,ip_num,_=get_ip_info(ip_file)
    num_datas=len(datas)
    step=num_datas//ip_num
    return datas[ip_index*step:(ip_index+1)*step] if ip_index<ip_num-1 else datas[ip_index*step:num_datas]

def get_init_file():
    MAX_TOKEN_LEN = cfg['sample']['max_len']
    big_dir = Path(cfg['data']['directory'])
    try:
        ip_index,_,_=get_ip_info(cfg['hosts'])
        print('SUNCCESS ->>>>>>>>>>>>')
    except:
        print(f"getting ip_index error, default to 0")
        ip_index=0
    DEFAULT_DIRECTORY=os.path.join(big_dir,f'part_{ip_index:02d}')
    print(DEFAULT_DIRECTORY)
    save_files_dir=os.path.join(DEFAULT_DIRECTORY,"save_files")
    os.makedirs(save_files_dir,exist_ok=True)
    OUTPUT_FILE = cfg['data']['output_base']
    TOKEN_INFO_FILE = cfg['data']['output_token']

    OUTPUT_FILE=os.path.join(save_files_dir,OUTPUT_FILE)
    TOKEN_INFO_FILE=os.path.join(save_files_dir,TOKEN_INFO_FILE)

    return TOKEN_INFO_FILE,MAX_TOKEN_LEN,save_files_dir,big_dir,DEFAULT_DIRECTORY
        
def get_num_boxs():
    pairs_dir=cfg['data']['directory']
    box_num=0
    sample_num=0
    for part_dir in tqdm(os.listdir(pairs_dir)):
        if not part_dir.endswith('wds'):
            file_path=os.path.join(pairs_dir,part_dir,'save_files','bins_boxs.pkl')
            with open(file_path, 'rb') as f:
                bin_boxes = pickle.load(f)
                box_num+=len(bin_boxes)
                sample_num+=sum([len(box) for box in bin_boxes])
    return box_num,sample_num

if __name__ == "__main__":
    file_path=os.path.join('/vlm/chengzheng/datasets/pdf_datas/part_00/save_files','bins_boxs.pkl')
    with open(file_path, 'rb') as f:
        bin_boxes = pickle.load(f)
        print(len(bin_boxes))