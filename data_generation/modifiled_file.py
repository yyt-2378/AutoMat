import os
from ase import io


def modify_cif_angles(cif_path, output_dir):
    # 读取 CIF 文件
    structure = io.read(cif_path)
    
    # 获取当前的晶胞信息（包括三个轴的长度和角度）
    cell = structure.get_cell()
    
    # 如果与 90.0 的误差小于等于 1.0，修改为 90.0
    if abs(alpha - 90.0) <= 1.0:
        # 只修改 alpha 角度
        cell[3] = 90.0
    if abs(beta - 90.0) <= 1.0:
        # 只修改 beta 角度
        cell[4] = 90.0
    if abs(gamma - 90.0) <= 1.0:
        # 只修改 gamma 角度
        cell[5] = 90.0
    
    # 更新晶胞
    structure.set_cell(cell, scale_atoms=False)

    # 保存修改后的CIF文件
    modified_file_path = os.path.join(output_dir, 'modified_' + os.path.basename(cif_file_path))
    io.write(modified_file_path, structure)
    print(f"Modified CIF file saved: {modified_file_path}")


def process_cif_files_in_directory(directory_path, output_dir):
    # 遍历目录中的所有CIF文件
    for filename in os.listdir(directory_path):
        if filename.endswith(".cif"):
            cif_file_path = os.path.join(directory_path, filename)
            modify_cif_angles(cif_file_path, output_dir)


# 调用函数，传入文件夹路径
directory_path = './data_generation/supercell_orthonalize_selected_cifs'  # 请将这里的路径修改为你的CIF文件夹路径
output_dir = './data_generation/modified_supercell_orthonalize_selected_cifs'
os.makedirs(output_dir, exist_ok=True)
process_cif_files_in_directory(directory_path, output_dir)
