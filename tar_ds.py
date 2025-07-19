import subprocess
import sys
import os
import glob
from typing import List

def tar_sequence(sequence_name: str) -> bool:
    """
    对指定序列执行tar打包命令
    
    Args:
        sequence_name: 序列名称
        
    Returns:
        bool: 执行成功返回True，失败返回False
    """
    cmd = [
        'tar', '-czvf', f'{sequence_name}.tgz', 
        f'{sequence_name}/processed/hamer',
        '--exclude=*/packed/*',
        '--exclude=*/processed/*',
        sequence_name
    ]
    
    try:
        print(f"正在打包序列: {sequence_name}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"成功打包: {sequence_name}.tgz")
        return True
    except subprocess.CalledProcessError as e:
        print(f"打包失败 {sequence_name}: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    except Exception as e:
        print(f"执行命令时发生错误 {sequence_name}: {e}")
        return False

def extract_sequence(tgz_file: str) -> bool:
    """
    解压指定的.tgz文件
    
    Args:
        tgz_file: .tgz文件路径
        
    Returns:
        bool: 解压成功返回True，失败返回False
    """
    # 从文件名提取序列名
    sequence_name = os.path.basename(tgz_file).replace('.tgz', '')
    
    # 创建序列目录
    if not os.path.exists(sequence_name):
        os.makedirs(sequence_name)
        print(f"创建目录: {sequence_name}")
    
    cmd = ['tar', '-xzvf', tgz_file, '-C', sequence_name]
    
    try:
        print(f"正在解压: {tgz_file}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"成功解压: {tgz_file} -> {sequence_name}/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"解压失败 {tgz_file}: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    except Exception as e:
        print(f"解压时发生错误 {tgz_file}: {e}")
        return False

def extract_all_tgz():
    """解压当前目录下所有.tgz文件"""
    tgz_files = glob.glob("*.tgz")
    
    if not tgz_files:
        print("当前目录下没有找到.tgz文件")
        return
    
    print(f"找到 {len(tgz_files)} 个.tgz文件")
    
    success_count = 0
    for tgz_file in tgz_files:
        if extract_sequence(tgz_file):
            success_count += 1
    
    print(f"\n完成！成功解压 {success_count}/{len(tgz_files)} 个文件")

def main():
    """主函数"""
    # 检查是否有解压命令参数
    if len(sys.argv) > 1 and sys.argv[1] in ['extract', 'x', '-x']:
        extract_all_tgz()
        return
    
    if len(sys.argv) > 1:
        # 从命令行参数获取序列名
        sequence_names = sys.argv[1:]
    else:
        # 交互式输入
        print("请输入序列名列表，每行一个，输入空行结束：")
        print("或者输入 'extract' 来解压当前目录下所有.tgz文件")
        sequence_names = []
        while True:
            name = input().strip()
            if not name:
                break
            if name in ['extract', 'x']:
                extract_all_tgz()
                return
            sequence_names.append(name)
    
    if not sequence_names:
        print("未提供序列名")
        return
    
    print(f"将处理 {len(sequence_names)} 个序列")
    
    success_count = 0
    for sequence_name in sequence_names:
        if tar_sequence(sequence_name):
            success_count += 1
    
    print(f"\n完成！成功打包 {success_count}/{len(sequence_names)} 个序列")

if __name__ == "__main__":
    main()
