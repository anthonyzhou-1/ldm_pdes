# Text Captioning

## Cylinder Flow 
The cylinder flow dataset is captioned according to the positions of the mesh points, which is able to extract the cylinder radius, position, and inlet velocity. The Reynolds number can also be approximated and inserted into a prompt. The provided data is already captioned but to re-caption a dataset:
```
python caption_cylinder.py 
```

## Smoke Buoyancy (NS2D)
The smoke buoyancy captioner relies on sk-image to detect edges in a given density field. The density field and its edges are passed to Claude Sonnet 3.5 along with a prompt to generate a caption. The provided data is already captioned by to re-caption a dataset:

```
python caption_ns2d.py 
```
Note that this requires an Anthropic API token and Anthropic API credits. 