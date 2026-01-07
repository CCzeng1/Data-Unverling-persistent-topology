import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

import pandas as pd
import csv

def read_cpr_data_vtau(file_path):
    """读取CPR数据文件，提取元数据和测量数据"""
    metadata = {}
    data_lines = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if ':' in line:
                    key, value_str = line[1:].strip().split(':', 1)
                    key = key.strip()
                    value_str = value_str.strip()
                    
                    # 特殊处理sweep_values参数
                    if key == 'sweep_values':
                        try:
                            value_arr = np.array(ast.literal_eval(value_str))
                            metadata['sweep_min'] = float(np.min(value_arr))
                            metadata['sweep_max'] = float(np.max(value_arr))
                            metadata['sweep_points'] = len(value_arr)
                            # 保留参数名参考
                            if 'sweep_parameter' in metadata:
                                metadata['sweep_parameter'] = metadata['sweep_parameter']
                        except:
                            metadata[key] = value_str
                        continue
                    
                    # 尝试解析数组
                    if value_str.startswith('['):
                        try:
                            value = np.array(ast.literal_eval(value_str))
                            metadata[key] = value
                            continue
                        except:
                            pass
                    
                    # 尝试数值转换
                    try:
                        if '.' in value_str or 'e' in value_str.lower():
                            value = float(value_str)
                        else:
                            value = int(value_str)
                    except ValueError:
                        value = value_str
                    
                    metadata[key] = value
            else:
                data_lines.append(line.strip())
    
    # 检测并跳过标题行
    if data_lines:
        try:
            float(data_lines[0].split(',')[0])
        except ValueError:
            data_lines = data_lines[1:]
    
    # 创建DataFrame
    df = pd.DataFrame([line.split(',') for line in data_lines if line], 
                      columns=[ 'v_tau','Phase', 'Current'])
    
    # 转换数据类型
    df = df.astype({
        'v_tau':float,
        'Phase': float,
        'Current': float
    })
    
    # 确保磁场方向正确
    df = df.sort_values(by='v_tau')
    
    return metadata, df


def read_ABS_data(file_path):
    """读取CPR数据文件，提取元数据和测量数据"""
    metadata = {}
    data_lines = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if ':' in line:
                    key, value_str = line[1:].strip().split(':', 1)
                    key = key.strip()
                    value_str = value_str.strip()
                    
                    # 特殊处理sweep_values参数
                    if key == 'sweep_values':
                        try:
                            value_arr = np.array(ast.literal_eval(value_str))
                            metadata['sweep_min'] = float(np.min(value_arr))
                            metadata['sweep_max'] = float(np.max(value_arr))
                            metadata['sweep_points'] = len(value_arr)
                            # 保留参数名参考
                            if 'sweep_parameter' in metadata:
                                metadata['sweep_parameter'] = metadata['sweep_parameter']
                        except:
                            metadata[key] = value_str
                        continue
                    
                    # 尝试解析数组
                    if value_str.startswith('['):
                        try:
                            value = np.array(ast.literal_eval(value_str))
                            metadata[key] = value
                            continue
                        except:
                            pass
                    
                    # 尝试数值转换
                    try:
                        if '.' in value_str or 'e' in value_str.lower():
                            value = float(value_str)
                        else:
                            value = int(value_str)
                    except ValueError:
                        value = value_str
                    
                    metadata[key] = value
            else:
                data_lines.append(line.strip())
    
    # 检测并跳过标题行
    if data_lines:
        try:
            float(data_lines[0].split(',')[0])
        except ValueError:
            data_lines = data_lines[1:]
    
    # 创建DataFrame
    df = pd.DataFrame([line.split(',') for line in data_lines if line], 
                      columns=['v_tau','Phase', 'Energy', 'DOS'])
    
    # 转换数据类型
    df = df.astype({
        'v_tau': float,
        'Phase': float,
        'Energy': float,
         'DOS': float
    })
    
    # 确保磁场方向正确
    df = df.sort_values(by='Phase')
    
    return metadata, df
                
def read_select_data(file_path, columns_to_read=None):
    """Load data with option to select specific columns
    
    Args:
        file_path (str): Path to data file
        columns_to_read (list, optional): List of column names to extract. 
            If None, reads all columns.
    
    Returns:
        tuple: (metadata dict, DataFrame with selected columns)
    """
    metadata = {}
    data_lines = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if ':' in line:
                    key, value_str = line[1:].strip().split(':', 1)
                    key = key.strip()
                    value_str = value_str.strip()
                    
                    # Parse arrays
                    if value_str.startswith('['):
                        try:
                            value = np.array(ast.literal_eval(value_str))
                            metadata[key] = value
                            continue
                        except:
                            pass
                    
                    # Convert numeric values
                    try:
                        if '.' in value_str or 'e' in value_str.lower():
                            value = float(value_str)
                        else:
                            value = int(value_str)
                    except ValueError:
                        value = value_str
                    
                    metadata[key] = value
            else:
                data_lines.append(line.strip())
    
    # Skip header if present
    header = None
    if data_lines:
        try:
            # Try to convert first value to float
            float(data_lines[0].split(',')[0])
        except ValueError:
            header = [col.strip() for col in data_lines[0].split(',')]
            data_lines = data_lines[1:]
    
    # Process data lines
    data = []
    for line in data_lines:
        if line:
            values = line.split(',')
            try:
                # Convert to float where possible
                row = []
                for v in values:
                    try:
                        row.append(float(v))
                    except ValueError:
                        row.append(v.strip())
                data.append(row)
            except:
                data.append([v.strip() for v in values])
    
    if not data:
        return metadata, pd.DataFrame()
    
    # Create DataFrame with or without header
    if header:
        df = pd.DataFrame(data, columns=header)
    else:
        df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(len(data[0]))])
    
    # Convert columns to numeric where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    
    # Select specific columns if requested
    if columns_to_read:
        available_cols = set(df.columns)
        selected_cols = [col for col in columns_to_read if col in available_cols]
        
        if not selected_cols:
            raise ValueError(f"None of the requested columns {columns_to_read} found in data")
        
        # Check for case sensitivity issues
        if len(selected_cols) < len(columns_to_read):
            missing = set(columns_to_read) - set(selected_cols)
            print(f"Warning: Some columns not found: {missing}. Using available columns.")
        
        df = df[selected_cols]
    
    return metadata, df


def read_select_data_cp(file_path, columns_to_read=None):
    """Load data with option to select specific columns
    
    Args:
        file_path (str): Path to data file
        columns_to_read (list, optional): List of column names to extract. 
            If None, reads all columns.
    
    Returns:
        tuple: (metadata dict, DataFrame with selected columns)
    """
    metadata = {}
    data_lines = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if ':' in line:
                    key, value_str = line[1:].strip().split(':', 1)
                    key = key.strip()
                    value_str = value_str.strip()
                    
                    # Parse arrays
                    if value_str.startswith('['):
                        try:
                            value = np.array(ast.literal_eval(value_str))
                            metadata[key] = value
                            continue
                        except:
                            pass
                    
                    # Convert numeric values
                    try:
                        if '.' in value_str or 'e' in value_str.lower():
                            value = float(value_str)
                        else:
                            value = int(value_str)
                    except ValueError:
                        value = value_str
                    
                    metadata[key] = value
            else:
                data_lines.append(line.strip())
    
    # Skip header if present
    header = None
    if data_lines:
        try:
            # Try to convert first value to float
            float(data_lines[0].split(',')[0])
        except ValueError:
            header = [col.strip() for col in data_lines[0].split(',')]
            data_lines = data_lines[1:]
    
    # Process data lines
    data = []
    for line in data_lines:
        if line:
            values = line.split(',')
            try:
                # Convert to float where possible
                row = []
                for v in values:
                    try:
                        row.append(float(v))
                    except ValueError:
                        row.append(v.strip())
                data.append(row)
            except:
                data.append([v.strip() for v in values])
    
    if not data:
        return metadata, pd.DataFrame()
    
    # Create DataFrame with or without header
    if header:
        df = pd.DataFrame(data, columns=header)
    else:
        df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(len(data[0]))])
    
    # Convert columns to numeric where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    
    # Select specific columns if requested
    if columns_to_read:
        available_cols = set(df.columns)
        selected_cols = [col for col in columns_to_read if col in available_cols]
        
        if not selected_cols:
            raise ValueError(f"None of the requested columns {columns_to_read} found in data")
        
        # Check for case sensitivity issues
        if len(selected_cols) < len(columns_to_read):
            missing = set(columns_to_read) - set(selected_cols)
            print(f"Warning: Some columns not found: {missing}. Using available columns.")
        
        df = df[selected_cols]
    
    return metadata, df

def read_cpr_data(file_path):
    """读取CPR数据文件，提取元数据和测量数据"""
    metadata = {}
    data_lines = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if ':' in line:
                    key, value_str = line[1:].strip().split(':', 1)
                    key = key.strip()
                    value_str = value_str.strip()
                    
                    # 特殊处理sweep_values参数
                    if key == 'sweep_values':
                        try:
                            value_arr = np.array(ast.literal_eval(value_str))
                            metadata['sweep_min'] = float(np.min(value_arr))
                            metadata['sweep_max'] = float(np.max(value_arr))
                            metadata['sweep_points'] = len(value_arr)
                            # 保留参数名参考
                            if 'sweep_parameter' in metadata:
                                metadata['sweep_parameter'] = metadata['sweep_parameter']
                        except:
                            metadata[key] = value_str
                        continue
                    
                    # 尝试解析数组
                    if value_str.startswith('['):
                        try:
                            value = np.array(ast.literal_eval(value_str))
                            metadata[key] = value
                            continue
                        except:
                            pass
                    
                    # 尝试数值转换
                    try:
                        if '.' in value_str or 'e' in value_str.lower():
                            value = float(value_str)
                        else:
                            value = int(value_str)
                    except ValueError:
                        value = value_str
                    
                    metadata[key] = value
            else:
                data_lines.append(line.strip())
    
    # 检测并跳过标题行
    if data_lines:
        try:
            float(data_lines[0].split(',')[0])
        except ValueError:
            data_lines = data_lines[1:]
    
    # 创建DataFrame
    df = pd.DataFrame([line.split(',') for line in data_lines if line], 
                      columns=['B', 'Phase', 'Current'])
    
    # 转换数据类型
    df = df.astype({
        'B': float,
        'Phase': float,
        'Current': float
    })
    
    # 确保磁场方向正确
    df = df.sort_values(by='B')
    
    return metadata, df

# ======================
# 画图参数设置函数
# ======================
def set_plotting_style():
    """设置全局绘图参数"""
    # 全局参数
    plt.rcParams['text.usetex'] = True # 使用LaTeX渲染文本
    plt.rcParams['font.family'] = 'serif'  # 使用衬线字体
    # mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"

    # # 字体大小设置
    # label_size = 20
    # title_size = 18
    # legend_size = 18
    # tick_size = 20
    # cbar_label_size = 18
    # cbar_tick_size = 18

    ## org set for figure_f
    
    # label_size = 16
    # title_size = 14
    # legend_size = 14
    # tick_size = 16  # 修改为16，与代码2一致
    # cbar_label_size = 14
    # cbar_tick_size = 14


    label_size = 14
    title_size = 14
    legend_size = 12
    tick_size = 12  # 修改为16，与代码2一致
    cbar_label_size = 12
    cbar_tick_size = 12
    
    # 设置全局字体大小
    mpl.rcParams['xtick.labelsize'] = tick_size
    mpl.rcParams['ytick.labelsize'] = tick_size
    mpl.rcParams['axes.labelsize'] = label_size
    mpl.rcParams['axes.titlesize'] = title_size
    mpl.rcParams['legend.fontsize'] = legend_size
    mpl.rcParams['figure.titlesize'] = title_size
    
    # 颜色设置
    colors = {
        'vtau_0.05': 'k',
        'vtau_0.75': '#E41A1C',  # 红色
        'vtau_0.85': '#377EB8',  # 蓝色
        'vtau_0.90': '#4DAF4A',  # 绿色
        'vtau_0.95': '#984EA3',  # 紫色
        'vtau_1.00': '#FF7F00',  # 橙色
        'grid': '#DDDDDD',
        'background': 'white',
        'text': 'black',
        'linecut_0': 'fuchsia',      # 第一个line cut颜色
        'linecut_1': 'aqua',      # 第二个line cut颜色
        'linecut_2': 'crimson',   # 
        'linecut_3': 'orange'      # 
        
    }
    
    # 线型设置
    line_styles = {
        'B=0.5': {'linestyle': ':', 'linewidth': 2.0},
        'B=2.0': {'linestyle': '-', 'linewidth': 3.0},
        'default': {'linestyle': '-', 'linewidth': 1.5}
    }
    
    # 透明度设置
    transparencies = {
        'legend': 0.9,
        'annotation': 0.8,
        'grid': 0.7
    }
    
    # 图例设置
    legend_settings = {
        'frameon': True,
        'fancybox': True,
        'framealpha': transparencies['legend'],
        'edgecolor': 'none'#'black'
    }
    
    # 刻度设置
    tick_settings = {
        'direction': 'in',
        'length': 6,
        'width': 1.5,
        'color': colors['text']
    }
    
    # 网格设置
    grid_settings = {
        'visible': False,
        'linestyle': '--',
        'linewidth': 0.5,
        'alpha': transparencies['grid'],
        'color': colors['grid']
    }
    
    # 返回所有设置
    return {
        'colors': colors,
        'line_styles': line_styles,
        'transparencies': transparencies,
        'legend_settings': legend_settings,
        'tick_settings': tick_settings,
        'grid_settings': grid_settings,
        'sizes': {
            'label': label_size,
            'title': title_size,
            'legend': legend_size,
            'tick': tick_size,
            'cbar_label': cbar_label_size,
            'cbar_tick': cbar_tick_size
        }
    }

def read_dc_data_used(file_path):
    """读取CPR数据文件，提取元数据和测量数据"""
    metadata = {}
    data_lines = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if ':' in line:
                    key, value_str = line[1:].strip().split(':', 1)
                    key = key.strip()
                    value_str = value_str.strip()
                    
                    # 特殊处理sweep_values参数
                    if key == 'sweep_values':
                        try:
                            value_arr = np.array(ast.literal_eval(value_str))
                            metadata['sweep_min'] = float(np.min(value_arr))
                            metadata['sweep_max'] = float(np.max(value_arr))
                            metadata['sweep_points'] = len(value_arr)
                            # 保留参数名参考
                            if 'sweep_parameter' in metadata:
                                metadata['sweep_parameter'] = metadata['sweep_parameter']
                        except:
                            metadata[key] = value_str
                        continue
                    
                    # 尝试解析数组
                    if value_str.startswith('['):
                        try:
                            value = np.array(ast.literal_eval(value_str))
                            metadata[key] = value
                            continue
                        except:
                            pass
                    
                    # 尝试数值转换
                    try:
                        if '.' in value_str or 'e' in value_str.lower():
                            value = float(value_str)
                        else:
                            value = int(value_str)
                    except ValueError:
                        value = value_str
                    
                    metadata[key] = value
            else:
                data_lines.append(line.strip())
    
    # 检测并跳过标题行
    if data_lines:
        try:
            float(data_lines[0].split(',')[0])
        except ValueError:
            data_lines = data_lines[1:]
    
    # 创建DataFrame
    df = pd.DataFrame([line.split(',') for line in data_lines if line], 
                      columns=[ 'Bias Voltage', 'Current','SidebandN'])
    
    # 转换数据类型
    df = df.astype({
        'Bias Voltage': float,
        'Current': float,
       'SidebandN': int
    })
    
    # 确保磁场方向正确
    df = df.sort_values(by='Bias Voltage')
    
    return metadata, df

def read_dc_data_waterfall(file_path, twoD_index= False,spectra_index =False):
    """读取CPR数据文件，提取元数据和测量数据"""
    metadata = {}
    data_lines = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if ':' in line:
                    key, value_str = line[1:].strip().split(':', 1)
                    key = key.strip()
                    value_str = value_str.strip()
                    
                    # 特殊处理sweep_values参数
                    if key == 'sweep_values':
                        try:
                            value_arr = np.array(ast.literal_eval(value_str))
                            metadata['sweep_min'] = float(np.min(value_arr))
                            metadata['sweep_max'] = float(np.max(value_arr))
                            metadata['sweep_points'] = len(value_arr)
                            # 保留参数名参考
                            if 'sweep_parameter' in metadata:
                                metadata['sweep_parameter'] = metadata['sweep_parameter']
                        except:
                            metadata[key] = value_str
                        continue
                    
                    # 尝试解析数组
                    if value_str.startswith('['):
                        try:
                            value = np.array(ast.literal_eval(value_str))
                            metadata[key] = value
                            continue
                        except:
                            pass
                    
                    # 尝试数值转换
                    try:
                        if '.' in value_str or 'e' in value_str.lower():
                            value = float(value_str)
                        else:
                            value = int(value_str)
                    except ValueError:
                        value = value_str
                    
                    metadata[key] = value
            else:
                data_lines.append(line.strip())
    
    # 检测并跳过标题行
    if data_lines:
        try:
            float(data_lines[0].split(',')[0])
        except ValueError:
            data_lines = data_lines[1:]
    
    # 创建DataFrame
    if twoD_index:
        if spectra_index:
            df = pd.DataFrame([line.split(',') for line in data_lines if line], 
                          columns=['v_tau','Magnetic Field', 'Energy', 'DOS'])
        else:
            df = pd.DataFrame([line.split(',') for line in data_lines if line], 
                          columns=['Magnetic Field', 'Bias Voltage', 'Current','SidebandN'])
    
    else:
        df = pd.DataFrame([line.split(',') for line in data_lines if line], 
                          columns=[ 'Bias Voltage', 'Current','SidebandN'])

    # 转换数据类型
    if spectra_index:
        df = df.astype({
            'Magnetic Field': float,
            'Energy': float,
           'DOS': float
            })
        # 确保磁场方向正确
        df = df.sort_values(by='Magnetic Field')
    else:
        df = df.astype({
            'Bias Voltage': float,
            'Current': float,
           'SidebandN': int
        })
        
        # 确保磁场方向正确
        df = df.sort_values(by='Bias Voltage')
        
    return metadata, df


def read_dc_data(file_path):
    """读取CPR数据文件，提取元数据和测量数据"""
    metadata = {}
    data_lines = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if ':' in line:
                    key, value_str = line[1:].strip().split(':', 1)
                    key = key.strip()
                    value_str = value_str.strip()
                    
                    # 特殊处理sweep_values参数
                    if key == 'sweep_values':
                        try:
                            value_arr = np.array(ast.literal_eval(value_str))
                            metadata['sweep_min'] = float(np.min(value_arr))
                            metadata['sweep_max'] = float(np.max(value_arr))
                            metadata['sweep_points'] = len(value_arr)
                            # 保留参数名参考
                            if 'sweep_parameter' in metadata:
                                metadata['sweep_parameter'] = metadata['sweep_parameter']
                        except:
                            metadata[key] = value_str
                        continue
                    
                    # 尝试解析数组
                    if value_str.startswith('['):
                        try:
                            value = np.array(ast.literal_eval(value_str))
                            metadata[key] = value
                            continue
                        except:
                            pass
                    
                    # 尝试数值转换
                    try:
                        if '.' in value_str or 'e' in value_str.lower():
                            value = float(value_str)
                        else:
                            value = int(value_str)
                    except ValueError:
                        value = value_str
                    
                    metadata[key] = value
            else:
                data_lines.append(line.strip())
    
    # 检测并跳过标题行
    if data_lines:
        try:
            float(data_lines[0].split(',')[0])
        except ValueError:
            data_lines = data_lines[1:]
    
    # 创建DataFrame
    df = pd.DataFrame([line.split(',') for line in data_lines if line], 
                      columns=['v_tau','Phase', 'Energy', 'DOS'])
    
    # 转换数据类型
    df = df.astype({
        'v_tau': float,
        'Phase': float,
        'Energy': float,
         'DOS': float
    })
    
    # 确保磁场方向正确
    df = df.sort_values(by='Phase')
    
    return metadata, df