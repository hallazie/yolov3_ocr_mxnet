from PIL import Image, ImageDraw, ImageFont
import json
import skimage

def rect():
	font = ImageFont.truetype('font/mriamc.ttf',16)
	for i in range(1):
		img = Image.open('data/imgs/%s.png'%i)
		draw = ImageDraw.Draw(img)
		with open('data/jsons/%s.json'%i, 'r') as jfile:
			jsn = json.load(jfile)
			for k in jsn:
				x0, y0, x1, y1 = jsn[k][0][0], jsn[k][0][1], jsn[k][1][0], jsn[k][1][1]
				draw.polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)], outline=(0,0,255))
				draw.text((x0,y0-16),k,font=font,fill=(0,0,255))
		img.save('data/%s.png'%i)

if __name__ == '__main__':
	rect()