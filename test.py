from PIL import Image,ImageFilter
import numpy as np
filename="test.jpg"
img=Image.open(filename)
height,width=img.size
center_ver=int(height/2)
center_hor=int(width/2)
img_p=np.array(img)[center_ver-64:center_ver+64,center_hor-64:center_hor+64,:]
img2_p=np.array([img_p[round(i/2)-1,:,:] for i in range(1,257)])
img3_p=np.array([img2_p[:,round(i/2)-1,:] for i in range(1,257)],dtype="uint8")
img3_p=img3_p.transpose(1,0,2)
img3=Image.fromarray(img3_p)
img3.show()
