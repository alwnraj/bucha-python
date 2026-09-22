"""Generate inspectable PNGs using NumPy and Pillow; no browser or GPU needed.

The preview renderer sorts faces by depth and samples texture color per face.
It is an approximate inspection view. Exported GLBs retain full original textures.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from segment_objects import accessor, load_geometry

BACKGROUND = '#f2f2ed'


def render(triangles, colors, size=(1300, 850), top=False, window=None, shade=True):
    image = Image.new('RGB', size, BACKGROUND)
    if not len(triangles):
        return image
    if top:
        right, up, direction = np.eye(3)
    else:
        direction = np.array([1., -1., .85])
        direction /= np.linalg.norm(direction)
        right = np.cross([0., 0., 1.], direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
    xy = np.stack([triangles @ right, triangles @ up], axis=-1)
    if window is None:
        lo = xy.min(axis=(0, 1))
        hi = xy.max(axis=(0, 1))
        extent = np.maximum(hi - lo, 1e-6)
        scale = min(size[0] * .90 / extent[0], size[1] * .90 / extent[1])
        xy = (xy - (lo + hi) / 2) * scale
        xy[..., 0] += size[0] / 2
        xy[..., 1] = size[1] / 2 - xy[..., 1]
    else:
        left, right_edge, bottom, top_edge = window
        xy[..., 0] = (xy[..., 0] - left) * size[0] / (right_edge - left)
        xy[..., 1] = (top_edge - xy[..., 1]) * size[1] / (top_edge - bottom)
    colors = np.asarray(colors, dtype=float)
    if shade:
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        light = np.array([.2, -.5, 1.])
        light /= np.linalg.norm(light)
        colors = colors * (.60 + .40 * np.abs(normals @ light))[:, None]
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    draw = ImageDraw.Draw(image)
    # Average-depth ordering is sufficient for this small-face inspection renderer.
    order = np.argsort(triangles.mean(axis=1) @ direction, kind='stable')
    polygons = xy.tolist()
    rgb = colors.tolist()
    for i in order:
        draw.polygon([tuple(p) for p in polygons[i]], fill=tuple(rgb[i]))
    return image


def texture_face_colors(doc, blob, chunks):
    images = {}
    result = []
    for chunk in chunks:
        primitive = chunk['primitive']
        material = doc['materials'][primitive['material']]['pbrMetallicRoughness']
        texture = doc['textures'][material['baseColorTexture']['index']]
        image_index = texture['source']
        if image_index not in images:
            view = doc['bufferViews'][doc['images'][image_index]['bufferView']]
            offset = view.get('byteOffset', 0)
            images[image_index] = np.asarray(Image.open(BytesIO(blob[offset:offset + view['byteLength']])).convert('RGB'))
        pixels = images[image_index]
        uv = accessor(doc, blob, primitive['attributes']['TEXCOORD_0'])[chunk['faces']].mean(axis=1)
        # This source's textures use repeat wrapping and glTF's top-left UV origin.
        uv %= 1.0
        u = np.minimum((uv[:, 0] * pixels.shape[1]).astype(int), pixels.shape[1] - 1)
        v = np.minimum((uv[:, 1] * pixels.shape[0]).astype(int), pixels.shape[0] - 1)
        result.append(pixels[v, u].astype(float) * np.array(material.get('baseColorFactor', [1, 1, 1, 1]))[:3])
    return np.concatenate(result)


def framed(image, title, objects, note):
    result = Image.new('RGB', (image.width, image.height + 166), BACKGROUND)
    result.paste(image, (0, 52))
    draw = ImageDraw.Draw(result)
    draw.text((26, 16), title, fill='#22342c', font=ImageFont.load_default(size=25))
    font = ImageFont.load_default(size=17)
    for index, obj in enumerate(objects):
        x = 26 + (index % 3) * (image.width // 3)
        y = image.height + 67 + (index // 3) * 29
        draw.rectangle((x, y + 2, x + 14, y + 17), fill=obj['color'])
        draw.text((x + 22, y), f"{obj['id']}: {obj['name']}", fill='#22342c', font=font)
    draw.text((26, result.height - 29), note, fill='#536359', font=ImageFont.load_default(size=15))
    return result


def generate_previews(input_glb, result_directory, annotations):
    result_directory = Path(result_directory)
    folder = result_directory / 'previews'
    folder.mkdir(exist_ok=True)
    manifest = json.loads((result_directory / 'manifest.json').read_text())
    cfg = json.loads(Path(annotations).read_text())
    objects = manifest['objects']
    doc, blob, chunks = load_geometry(input_glb)
    triangles = np.concatenate([c['tri'] for c in chunks])
    with np.load(result_directory / 'face-labels.npz') as data:
        labels = data['object_id']
    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    for obj in objects:
        color = tuple(int(obj['color'][i:i + 2], 16) for i in (1, 3, 5))
        colors[labels == obj['id']] = color
    note = 'Provisional labels. Source +Z up; scale unverified. NumPy + Pillow render.'
    framed(render(triangles, colors), 'Bucha scan: separate object labels', objects, note).save(folder / 'scene-labeled.png')
    framed(render(triangles, colors, top=True, shade=False), 'Top-down labels: common world XY coordinates', objects, note).save(folder / 'top-labeled.png')
    sampled_texture = texture_face_colors(doc, blob, chunks)
    framed(render(triangles, sampled_texture, shade=False), 'Original texture: approximate per-face preview', objects,
           'PNG uses per-face texture samples. Exported GLBs keep the full original UV texture mapping.').save(folder / 'scene-textured.png')
    # Matches the pixel-to-world convention stored in annotations.json exactly.
    annotation_image = render(triangles, sampled_texture, size=(1240, 1640), top=True,
                              window=(-5, 26, -14, 27), shade=False)
    draw = ImageDraw.Draw(annotation_image)
    font = ImageFont.load_default(size=25)
    for obj in cfg['objects']:
        for polygon in obj['polygons_pixels']:
            draw.line([tuple(p) for p in polygon + [polygon[0]]], fill=obj['color'], width=3)
            center = np.asarray(polygon).mean(axis=0)
            draw.text(tuple(center), str(obj['id']), fill=obj['color'], font=font, stroke_width=2, stroke_fill='white')
    annotation_image.save(folder / 'annotation-map.png')
    sheet = Image.new('RGB', (1320, 1370), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 18), 'Six separate assets: provisional object labels', fill='#22342c', font=ImageFont.load_default(size=25))
    for index, obj in enumerate(objects):
        selected = labels == obj['id']
        x, y = (index % 2) * 660, 64 + (index // 2) * 430
        sheet.paste(render(triangles[selected], colors[selected], size=(660, 350)), (x, y))
        draw.text((x + 22, y + 355), obj['name'], fill='#22342c', font=ImageFont.load_default(size=22))
        draw.text((x + 22, y + 386), f"{obj['triangles']:,} faces | ID {obj['id']}", fill='#536359', font=ImageFont.load_default(size=17))
    sheet.save(folder / 'objects-preview.png')
    return ['previews/' + name for name in ['scene-labeled.png', 'top-labeled.png', 'scene-textured.png',
                                          'annotation-map.png', 'objects-preview.png']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_glb')
    parser.add_argument('result_directory')
    parser.add_argument('--annotations', default=str(Path(__file__).with_name('annotations.json')))
    args = parser.parse_args()
    for filename in generate_previews(args.input_glb, args.result_directory, args.annotations):
        print(filename)
