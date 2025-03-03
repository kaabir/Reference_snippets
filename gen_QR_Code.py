import qrcode
from PIL import Image

def generate_qr(data, filename="qr_black.png", size=10, border=4, background="white", transparent=True):
    """

    """
    
    # Create QR Code
    qr = qrcode.QRCode(
        version=1,  # Adjust for complexity
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=size,
        border=border,
    )
    
    qr.add_data(data)
    qr.make(fit=True)
    
    # Create image
    qr_image = qr.make_image(fill="black", back_color=background).convert("RGBA")

    if transparent:
        # Convert white pixels to transparent
        qr_image = qr_image.convert("RGBA")
        datas = qr_image.getdata()
        new_data = []
        
        for item in datas:
            if item[:3] == (0, 0, 0):  # Keep black pixels
                new_data.append((0, 0, 0, 255))
            else:  # Make white pixels transparent
                new_data.append((0, 0, 0, 0))
        
        qr_image.putdata(new_data)
    
    # Save the QR Code
    qr_image.save(filename)
    print(f"QR Code saved as {filename}")


generate_qr("https://", filename="qr_transparent.png", transparent=False)  # Transparent background
