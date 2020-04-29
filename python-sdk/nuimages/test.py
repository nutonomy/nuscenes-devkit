from nuimages import NuImages
import matplotlib.pyplot as plt

nuim = NuImages(verbose=True, lazy=True)

print(nuim.category[0])
print(nuim.object_ann[0])

im = nuim.explorer.render_image(nuim.image[0]['token'])
plt.show()