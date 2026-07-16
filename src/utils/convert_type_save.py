import numpy as np
from scipy.io import loadmat
import pickle


data_mat_path = r"./ground_truth_python\ground_truth_matlab\Niter1000.mat"
data = loadmat(data_mat_path)


def convert_void_to_dict(void_array):
    """
    Convert a numpy.void object into a dictionary by iterating over its fields.
    """
    result = {}
    for field in void_array.dtype.names:
        # Directly access each field of the numpy.void object.
        value = void_array[field]
        # Check if the field itself contains a structured type.
        if isinstance(value, np.void):
            result[field] = convert_void_to_dict(value)
        else:
            result[field] = value
    return result


def convert_nested_arrays_to_dict(data):
    """
    Convert a dictionary containing numpy arrays and possibly numpy.void objects into a more straightforward dict structure.
    """
    result_dict = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            unpacked = {}
            # Check if the first element is a numpy.void, indicating a structured array.
            if value.size > 0 and isinstance(value[0, 0], np.void):
                for item in np.nditer(value, flags=['refs_ok']):
                    # item[()] is used to get the actual object referenced by item.
                    if isinstance(item[()], np.void):
                        unpacked_dict = convert_void_to_dict(item[()])
                        unpacked.update(unpacked_dict)
                result_dict[key] = unpacked
            else:
                result_dict[key] = value
        else:
            result_dict[key] = value
    return result_dict


# # 转换数据
# converted_data = convert_nested_arrays_to_dict(data)
#
# # 保存数据到pickle文件
# with open('data.pkl', 'wb') as file:
#     pickle.dump(converted_data, file)

# 从pickle文件加载数据
with open('data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# 输出结果，以查看转换后的字典结构
for key, value in loaded_data.items():
    if isinstance(value, list):
        print(f"{key}: List of {len(value)} items")
    elif isinstance(value, dict):
        print(f"{key}: Dictionary")
    else:
        print(f"{key}: {value.shape}, dtype={value.dtype}")
