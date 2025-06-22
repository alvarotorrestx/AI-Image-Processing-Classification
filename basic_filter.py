from PIL import Image, ImageFilter, ImageOps, ImageDraw
import matplotlib.pyplot as plt

def save_filtered_image(img, filter_name, filename):
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    print(f"✅ Processed image saved as '{filename}'.")

def apply_multiple_filters(image_path):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))

        # Gaussian Blur
        img_blur = img_resized.filter(ImageFilter.GaussianBlur(radius=2))
        save_filtered_image(img_blur, "Blur", "filtered_blur.png")

        # Edge Detection
        img_edges = img_resized.filter(ImageFilter.FIND_EDGES)
        save_filtered_image(img_edges, "Edges", "filtered_edges.png")

        # Sharpen
        img_sharp = img_resized.filter(ImageFilter.SHARPEN)
        save_filtered_image(img_sharp, "Sharpen", "filtered_sharpen.png")

        # Emboss
        img_emboss = img_resized.filter(ImageFilter.EMBOSS)
        save_filtered_image(img_emboss, "Emboss", "filtered_emboss.png")

        # Batman Beyond infrared scanner mode
        bb_vision = apply_batman_beyond_vision(img_resized)
        save_filtered_image(bb_vision, "Batman Beyond Vision", "filtered_bbvision.png")

    except Exception as e:
        print(f"❌ Error processing image: {e}")

def apply_batman_beyond_vision(img):
    gray = ImageOps.grayscale(img)

    red_overlay = Image.new("RGB", gray.size, (255, 0, 0))
    gray_rgb = Image.merge("RGB", (gray, gray, gray))
    infrared = Image.blend(gray_rgb, red_overlay, alpha=0.6)

    draw = ImageDraw.Draw(infrared)
    for y in range(0, infrared.height, 4):
        draw.line((0, y, infrared.width, y), fill=(255, 0, 0), width=1)

    return infrared


def apply_cyberpunk_filter_with_glitch(img):
    from PIL import ImageChops, ImageEnhance, ImageDraw, ImageFont

    img = img.convert("RGB").resize((128, 128))

    r, g, b = img.split()
    r_shifted = ImageChops.offset(r, 2, 0)
    g_shifted = ImageChops.offset(g, -2, 0)
    b_shifted = ImageChops.offset(b, 0, 2)
    img_glitch = Image.merge("RGB", (r_shifted, g_shifted, b_shifted))

    r, g, b = img_glitch.split()
    r = r.point(lambda i: i * 1.3)
    b = b.point(lambda i: i * 1.5)
    g = g.point(lambda i: i * 0.7)
    img_glitch = Image.merge("RGB", (r, g, b))

    draw = ImageDraw.Draw(img_glitch)
    for y in range(0, img_glitch.height, 3):
        draw.line((0, y, img_glitch.width, y), fill=(0, 0, 0, 100))

    img_glitch = ImageEnhance.Contrast(img_glitch).enhance(1.6)
    img_glitch = ImageEnhance.Color(img_glitch).enhance(2.5)

    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except:
        font = ImageFont.load_default()
    draw.text((4, 4), "CYBERPUNK GLITCH", font=font, fill=(255, 0, 255))

    return img_glitch



if __name__ == "__main__":
    image_path = "kami.png"
    apply_multiple_filters(image_path)

    try:
        img = Image.open(image_path)
        cyberpunk_glitch_img = apply_cyberpunk_filter_with_glitch(img)
        save_filtered_image(cyberpunk_glitch_img, "Cyberpunk Glitch", "filtered_cyberpunk_glitch.png")
        print("🟣 Cyberpunk Glitch filter applied ➜ 'filtered_cyberpunk_glitch.png'")
    except Exception as e:
        print(f"❌ Error applying Cyberpunk Glitch filter: {e}")
