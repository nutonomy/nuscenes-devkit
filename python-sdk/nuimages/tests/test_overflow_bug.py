from nuimages.nuimages import NuImages

tokens = [
    '6b17bab7b6f849abb7bbae05806eb2b9'  # Math overflow bug.
]

nuim = NuImages(version='v1.0-val')
for token in tokens:
    nuim.render_depth(tokens)
