import os
image_folder = "/home/meth/code/mlopsfin/data_pipeline/flow/images"  
output_txt = "/home/meth/code/mlopsfin/data_pipeline/flow/image.txt"    
image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
image_names = [f"train/{os.path.splitext(f)[0]}" for f in image_files]
with open(output_txt, "w") as f:
    for name in image_names:
        f.write(name + "\n")

print(f"Creating {output_txt} successfully {len(image_names)} rows.")
