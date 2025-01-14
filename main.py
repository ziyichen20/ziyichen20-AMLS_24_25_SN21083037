import subprocess

print(f"CNN for breastmnist---------------------------")
# Path to the script you want to run
script_path = r"E:\ML_CW\A\custom CNN test breast.py"
try:
    # Use subprocess to execute the script
    subprocess.run(["python", script_path], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred while running the script: {e}")
except FileNotFoundError:
    print(f"The specified script was not found: {script_path}")
    
print(f"transfer learning for breastmnist---------------------------")
# Path to the script you want to run
script_path = r"E:\ML_CW\A\transfer learning test breast.py"
try:
    # Use subprocess to execute the script
    subprocess.run(["python", script_path], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred while running the script: {e}")
except FileNotFoundError:
    print(f"The specified script was not found: {script_path}")

print(f"Resnet18 for bloodmnist---------------------------")
# Path to the script you want to run
script_path = r"E:\ML_CW\B\Resnet18 test blood.py"
try:
    # Use subprocess to execute the script
    subprocess.run(["python", script_path], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred while running the script: {e}")
except FileNotFoundError:
    print(f"The specified script was not found: {script_path}")
    
    
print(f"Custom CNN for bloodmnist---------------------------")
script_path = r"E:\ML_CW\B\customCNN test blood.py"

try:
    # Use subprocess to execute the script
    subprocess.run(["python", script_path], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred while running the script: {e}")
except FileNotFoundError:
    print(f"The specified script was not found: {script_path}")
