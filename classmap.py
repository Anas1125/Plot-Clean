import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
LINE,BLUE,RED=1,2,3
def class_map(path_or_im):
    """uint8 map: 0 paper, 1 neutral dark ink (lines), 2 blue ink (text), 3 red ink (labels)."""
    im = Image.open(path_or_im) if isinstance(path_or_im,str) else path_or_im
    def cls(rgb):
        rgb = rgb.astype(np.int32)
        r,g,b = rgb[...,0],rgb[...,1],rgb[...,2]
        lum = (r*299+g*587+b*114)//1000
        out = np.zeros(r.shape,np.uint8)
        ink = lum < 205
        blue = ink & ((b-r) > 40) & ((b-g) > 25)
        red = ink & ((r-b) > 40) & ((r-g) > 25)
        out[ink] = LINE
        out[blue] = BLUE; out[red] = RED
        return out
    if im.mode=='P':
        pal = np.array(im.getpalette()[:768]+[0]*(768-len(im.getpalette()[:768])),np.uint8).reshape(-1,3)[:256]
        lut = cls(pal[None,:,:])[0]
        return lut[np.array(im)]
    H = im.size[1]; out = np.zeros((H,im.size[0]),np.uint8)
    for y in range(0,H,1024):
        out[y:y+1024] = cls(np.array(im.crop((0,y,im.size[0],min(H,y+1024))).convert('RGB')))
    return out
