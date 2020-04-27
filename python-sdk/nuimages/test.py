from nuimages import NuImages

nuim = NuImages(verbose=True, lazy=False)

print(nuim.category[0])
print(nuim.object_ann[0])