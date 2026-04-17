from PIL import Image
import argparse
import sys

def convert_ppm_to_image(input_path, output_path):
    try:
        # 打开 PPM 文件
        img = Image.open(input_path)
        
        # 保存为目标格式（根据输出文件后缀自动决定）
        img.save(output_path)
        
        print(f"✅ 转换成功！")
        print(f"输入文件: {input_path}")
        print(f"输出文件: {output_path}")
        print(f"图片尺寸: {img.size[0]} x {img.size[1]}")
        print(f"格式: {img.format} → {output_path.split('.')[-1].upper()}")
        
    except FileNotFoundError:
        print(f"❌ 错误：找不到输入文件 '{input_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 转换失败：{e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 PPM 图像转换为 JPG/PNG 等常见格式")
    parser.add_argument("input", help="输入的 PPM 文件路径")
    parser.add_argument("output", help="输出的图片文件路径（支持 .jpg, .png, .bmp 等）")
    
    args = parser.parse_args()
    
    convert_ppm_to_image(args.input, args.output)