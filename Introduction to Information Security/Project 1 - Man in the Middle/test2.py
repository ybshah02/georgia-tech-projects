def ascii_to_binary(ascii_dump):
    # Remove newlines and spaces
    ascii_dump = ''.join(ascii_dump.split())
    
    # Convert each pair of hex characters to a byte
    binary_data = bytes.fromhex(ascii_dump)
    
    return binary_data

# Your ASCII dump
ascii_dump = """
610d0d0a00000000ff6ea6662c020000e30000000000000000000000000000000004000000400000007352000000640064016c005a0065016402830101006502640383015a036500a0046503a005a100a1015a066506a007a1005a086500a0046404a1015a096509a007a1005a0a650164056508650a170083020100640153002906e9000000004e7a1b3d3d3d3d20596f7520666f756e6420466c6167203421203d3d3d3d7a12456e74657220796f75722047544944203a2073140000004353363033352d6e4f76342d42346e64336972347a13436f6d62696e656420686173682020203a2020290b5a07686173686c6962da057072696e74da05696e707574da01785a06736861323536da06656e636f64655a0b686173685f6f626a6563745a096865786469676573745a076865785f6469675a0c686173685f6f626a656374325a096865785f6469673232a90072060000007206000000fa08666c6167342e7079da083c6d6f64756c653e0b000000730e0000000802080108020e0108010a010801
"""

binary_data = ascii_to_binary(ascii_dump)

# Write to a file
with open('output.pyc', 'wb') as f:
    f.write(binary_data)

import uncompyle6
import io
import sys

def decompile_pyc(filename):
    with open(filename, 'rb') as pyc_file:
        # Read the content of the file
        content = pyc_file.read()
    
    # Create a file-like object from the content
    bytecode = io.BytesIO(content)
    
    # Redirect stdout to capture the decompiled code
    sys.stdout = io.StringIO()
    
    try:
        # Decompile the bytecode
        uncompyle6.decompile_file(bytecode, sys.stdout)
        
        # Get the decompiled code
        decompiled_code = sys.stdout.getvalue()
    finally:
        # Reset stdout
        sys.stdout = sys.__stdout__
    
    return decompiled_code

# Usage
filename = 'output.pyc'  # Replace with your actual filename
decompiled_code = decompile_pyc(filename)
print(decompiled_code)