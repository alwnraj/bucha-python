"""Run segmentation, object export, collision export and PNG previews from Python."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root / 'config.json')
    parser.add_argument('--input', type=Path, help='Override the source GLB path')
    parser.add_argument('--output', type=Path, help='Override the result directory')
    parser.add_argument('--annotations', type=Path, help='Override the annotation file')
    parser.add_argument('--no-previews', action='store_true')
    args = parser.parse_args()
    try:
        import numpy
        import PIL
        from segment_objects import segment
        from collision import export_collisions, raycast_scene
        from render_previews import generate_previews
    except ModuleNotFoundError as error:
        parser.error(f'{error}. Run setup_env.py, then select the .venv interpreter in VS Code.')
    try:
        config_path = args.config.resolve()
        config = json.loads(config_path.read_text())
        source = (args.input or config_path.parent / config['input_glb']).resolve()
        output = (args.output or config_path.parent / config['output_directory']).resolve()
        annotations = (args.annotations or config_path.parent / config['annotations']).resolve()
        if not source.is_file():
            raise ValueError(f'Source GLB not found: {source}')
        started = perf_counter()
        print('[1/4] Segmenting the supplied scan and exporting textured objects...', flush=True)
        manifest = segment(source, output, annotations)
        for obj in manifest['objects']:
            print(f"  {obj['id']}  {obj['name']}: {obj['triangles']:,} faces", flush=True)
        print('[2/4] Exporting one collision surface per object (NPZ + OBJ)...', flush=True)
        export_collisions(source, output)
        previews = []
        if not args.no_previews and config.get('generate_previews', True):
            print('[3/4] Rendering PNG previews in Python...', flush=True)
            previews = generate_previews(source, output, annotations)
        else:
            print('[3/4] Previews skipped.', flush=True)
        print('[4/4] Checking an example collision ray and writing the run report...', flush=True)
        ray = {'origin': [10, 0, 20], 'direction': [0, 0, -1]}
        ray['hit'] = raycast_scene(output / 'collision-scene.json', ray['origin'], ray['direction'])
        shutil.copyfile(annotations, output / 'annotations-used.json')
        shutil.copyfile(root / 'source-license.json', output / 'source-license.json')
        elapsed = perf_counter() - started
        report = {'input': str(source), 'source_sha256': manifest['source_sha256'],
                  'total_faces': sum(obj['triangles'] for obj in manifest['objects']),
                  'python': sys.version.split()[0], 'numpy': numpy.__version__, 'pillow': PIL.__version__,
                  'elapsed_seconds': round(elapsed, 2), 'previews': previews, 'example_ray': ray,
                  'validation': manifest['validation'], 'semantic_accuracy_validated': False,
                  'unity_and_ml2_executed': False}
        (output / 'run-report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'\nDone in {elapsed:.1f}s. Results: {output}', flush=True)
        print('Labels are scan-specific and provisional; collision surfaces are not a navigation planner.')
    except (ValueError, OSError, KeyError, AssertionError) as error:
        parser.exit(1, f'Pipeline stopped: {error}\n')


if __name__ == '__main__':
    main()
