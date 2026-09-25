import re

def process_kwargs(call_str):
    # Remove unsupported kwargs completely
    kwargs_to_remove = [
        'activebackground', 'activeforeground', 
        'highlightbackground', 'highlightcolor', 'highlightthickness', 
        'relief', 'bd', 'borderwidth', 'orient', 'length', 
        'showvalue', 'resolution', 'troughcolor', 'sliderrelief', 'sliderlength',
        'padx', 'pady'
    ]
    
    for kwarg in kwargs_to_remove:
        # Match `kwarg=...`
        pattern = r'\b' + kwarg + r'\s*=\s*(?:(?:\'[^\']*\')|(?:"[^\"]*")|([^,)\n]+))\s*,?\s*'
        call_str = re.sub(pattern, '', call_str)
        
    # Clean up trailing comma before closing parenthesis
    call_str = re.sub(r',\s*\)', ')', call_str)
    
    return call_str

def migrate_to_ctk(filepath, out_filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 1. Imports
    if 'import customtkinter as ctk' not in content:
        content = content.replace('import tkinter as tk', 'import customtkinter as ctk\nimport tkinter as tk')
    
    # Global renames for CustomTkinter support
    content = content.replace('.config(', '.configure(')
    content = re.sub(r'\bbg\s*=', 'fg_color=', content)
    content = re.sub(r'\bfg\s*=', 'text_color=', content)
    
    # 2. Process all tk widget constructors and .configure calls carefully
    # Find `tk.Frame(`, `tk.Label(`, etc., and `.configure(`
    search_strings = [f'tk.{w}(' for w in ['Tk', 'Frame', 'Label', 'Button', 'Scale', 'Entry']]
    search_strings.append('.configure(')
    
    for search_str in search_strings:
        idx = 0
        while True:
            idx = content.find(search_str, idx)
            if idx == -1:
                break
            
            # Found a call, find the balancing parenthesis
            start_idx = idx + len(search_str) - 1  # points to the '('
            paren_count = 0
            end_idx = -1
            for i in range(start_idx, len(content)):
                if content[i] == '(':
                    paren_count += 1
                elif content[i] == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        end_idx = i
                        break
            
            if end_idx != -1:
                call_str = content[idx:end_idx+1]
                
                # Remove kwargs
                new_call_str = process_kwargs(call_str)
                
                # Change prefix to ctk if it's a constructor
                if search_str.startswith('tk.'):
                    w_type = search_str[3:-1]
                    if w_type == 'Tk': new_call_str = new_call_str.replace('tk.Tk', 'ctk.CTk')
                    elif w_type == 'Frame': new_call_str = new_call_str.replace('tk.Frame', 'ctk.CTkFrame')
                    elif w_type == 'Label': new_call_str = new_call_str.replace('tk.Label', 'ctk.CTkLabel')
                    elif w_type == 'Button': new_call_str = new_call_str.replace('tk.Button', 'ctk.CTkButton')
                    elif w_type == 'Scale': new_call_str = new_call_str.replace('tk.Scale', 'ctk.CTkSlider')
                    elif w_type == 'Entry': new_call_str = new_call_str.replace('tk.Entry', 'ctk.CTkEntry')
                
                # Replace in content
                content = content[:idx] + new_call_str + content[end_idx+1:]
                # Advance idx to end of the new call
                idx = idx + len(new_call_str)
            else:
                idx += 1
                
    # 4. Special cases
    content = re.sub(r'style\.theme_use\(\s*\'clam\'\s*\)', 'pass', content)
    
    with open(out_filepath, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    migrate_to_ctk('/home/iit/rt_test/src/robot_control/robot_control/robot_gui_test.py', '/home/iit/rt_test/src/robot_control/robot_control/robot_gui_test_ctk.py')
