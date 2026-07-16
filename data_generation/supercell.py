import os
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter


# 指定存放 CIF 文件的文件夹路径（请替换为实际路径）
folder_path = './data_generation/selected_cif_files'

# 指定输出超胞 CIF 文件存放的文件夹路径，若不存在则创建
output_folder = './data_generation/supercell_selected_cifs'
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有 CIF 文件
for filename in os.listdir(folder_path):
    if filename.lower().endswith('.cif'):
        file_path = os.path.join(folder_path, filename)

        # 获取文件大小（单位：KB）
        file_size_kb = os.path.getsize(file_path) / 1024.0

        # 根据文件大小判断扩胞倍数
        if file_size_kb <= 1.0:
            Nx, Ny = 16, 16
        elif file_size_kb <= 2:
            Nx, Ny = 12, 12
        elif file_size_kb <= 3:
            Nx, Ny = 8, 8
        else:
            # 如果大于6KB，可以自行调整，示例默认为2×2
            Nx, Ny = 4, 4

        Nz = 1  # z方向保持单层

        # 读取 CIF 文件，生成原始结构
        original_structure = Structure.from_file(file_path)

        # 生成超胞
        supercell = original_structure * (Nx, Ny, Nz)

        # 构造输出文件名，例如 "原文件名_supercell_16x16x1.cif"
        base_name = os.path.splitext(filename)[0]
        output_file = f"{base_name}_supercell_{Nx}x{Ny}x{Nz}.cif"
        output_path = os.path.join(output_folder, output_file)

        # 将超胞写入新的 CIF 文件
        cif_writer = CifWriter(supercell)
        cif_writer.write_file(output_path)

        print(f"文件 {filename} 大小约: {file_size_kb:.2f} KB，采用 {Nx}×{Ny}×{Nz} 扩胞。")
        print(f"超胞 CIF 文件已保存至: {output_path}\n")

