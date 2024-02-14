import os
import re
def format_value(value):
    # Attempt to format the floating point value to scientific notation
    try:
        # Ensure the value is treated as a float, then format
        formatted_value = "{:.1e}".format(float(value))
        # Remove the .0 for consistency with requested format
        formatted_value = formatted_value.replace('.0e', 'e')
    except ValueError as e:
        print(f"Error formatting value {value}: {e}")
        formatted_value = value
    return formatted_value

def rename_directories(root_path):
    for root, dirs, files in os.walk(root_path, topdown=False):
        if "MultiNLI" in root or "clip" not in root:
            continue
        print(f"Processing directory: {root}")
        if "grad_alpha" in root and "hess_beta" in root:  # Check if both patterns are present
            # Extract grad_alpha and hess_beta values using regex
            grad_alpha_match = re.search("grad_alpha_([0-9\.e-]+)", root)
            hess_beta_match = re.search("hess_beta_([0-9\.e-]+)", root)
            if grad_alpha_match and hess_beta_match:
                grad_alpha_value = format_value(grad_alpha_match.group(1))
                hess_beta_value = format_value(hess_beta_match.group(1))
                # Construct new directory name by replacing old values with formatted values
                new_dir_name = re.sub("grad_alpha_[0-9\.e-]+", f"grad_alpha_{grad_alpha_value}", root)
                new_dir_name = re.sub("hess_beta_[0-9\.e-]+", f"hess_beta_{hess_beta_value}", new_dir_name)
                if new_dir_name != root:
                    print(f"Renaming '{root}' to '{new_dir_name}'")
                    os.rename(root, new_dir_name)
                else:
                    print(f"Directory name is already formatted: {root}")
            else:
                print(f"Could not find grad_alpha or hess_beta values in: {root}")
        else:
            print(f"No matching pattern in: {root}")  # Debug print
def main():
    # Specify the root path where your directories are located
    root_path = './logs'
    rename_directories(root_path)

if __name__ == "__main__":
    main()