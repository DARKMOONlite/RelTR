# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import argparse
import cProfile
import pstats
from PIL import Image
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import torch
import torchvision.transforms as T
from tqdm import tqdm
from models import build_model

CLASSES = ['N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
           'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
           'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
           'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
           'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
           'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
           'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
           'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
           'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
           'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
           'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
           'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
           'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
           'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
               'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
               'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
               'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
               'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
               'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def load_model(args):
    """Load model once and return it."""
    model, _, _ = build_model(args)
    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def infer(model, args,image):
    """Run inference on a single image using an already-loaded model."""
    im = Image.open(image).convert('RGB')
    img = transform(im).unsqueeze(0)
    outputs = model(img)

    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
    keep = torch.logical_and(probas.max(-1).values > 0.3,
                             torch.logical_and(probas_sub.max(-1).values > 0.3,
                                               probas_obj.max(-1).values > 0.3))

    sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
    obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    indices = torch.argsort(-probas[keep_queries].max(-1)[0]
                            * probas_sub[keep_queries].max(-1)[0]
                            * probas_obj[keep_queries].max(-1)[0])[:10]
    keep_queries = keep_queries[indices]

    return im, img, probas, probas_sub, probas_obj, keep_queries, indices, sub_bboxes_scaled, obj_bboxes_scaled, keep


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    # image path
    parser.add_argument('--img_path', type=str, default=None,
                        help="Path of the test image")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='ckpt/checkpoint0149_oi.pth', help='resume from checkpoint')
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")


    # distributed training parameters
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser

def bbox_iou(a, b):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    
    return abs(inter / (3*inter-area_a-area_b))  # gives a range of [0, 1] but is more forgiving for small boxes that don't perfectly align

REQUIRED_IOU_THRESHOLD = 0.3


def average_bbox(b1, b2):
    """Average two [x1,y1,x2,y2] boxes."""
    return [(b1[0] + b2[0]) / 2, (b1[1] + b2[1]) / 2, (b1[2] + b2[2]) / 2, (b1[3] + b2[3]) / 2]

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

def collect_images(root_dir, depth):
    """Collect image paths up to `depth` folder levels deep (1 = root only)."""
    images = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        rel = os.path.relpath(dirpath, root_dir)
        level = 0 if rel == '.' else rel.count(os.sep) + 1
        if level >= depth:
            dirnames.clear()  # don't recurse beyond requested depth
        for f in sorted(filenames):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                images.append(os.path.join(dirpath, f))
    return sorted(images)


def main(model, args,image):
    im, img, probas, probas_sub, probas_obj, keep_queries, indices, sub_bboxes_scaled, obj_bboxes_scaled, keep = infer(model, args,image)

    # Stores (label, bbox_tensor) for each registered object
    object_registry = [] # List of tuples: (label, bbox_list) for each registered object
    objects = []
    attributes = []
    relationships = {}  # Use dict to prevent duplicates: key = (subject, predicate, object)



    def get_or_create_object(label:str, bbox:torch.Tensor):
        bbox_list = bbox.tolist()
        
        for idx, (reg_label, reg_bbox) in enumerate(object_registry):
            if reg_label == label:
                iou = bbox_iou(bbox_list, reg_bbox)
                print(f"    Comparing with existing {reg_label} (object {idx})")
                print(f"      Existing bbox: {reg_bbox}")
                print(f"      IoU: {iou:.4f} (threshold: {REQUIRED_IOU_THRESHOLD})")
                if iou > REQUIRED_IOU_THRESHOLD:
                    print(f"    ✓ Reusing existing object {idx}")
                    # object_registry[idx] = (reg_label, average_bbox(reg_bbox, bbox_list))  # Update with average bbox
                    return idx
                else:
                    print(f"    ✗ IoU too low, will create new object")
        
        object_idx = len(objects)
        print(f"    → Creating new object {object_idx}")
        objects.append({"name": label,"bbox": [round(coord, 2) for coord in bbox_list]})
        object_registry.append((label, bbox_list))
        return object_idx




    for idx in keep_queries:
        # Find the correct position in the filtered bbox arrays
        # by counting how many True values appear before this index in 'keep'
        idx_in_kept = keep[:idx].sum().item()
        sub_idx:int = get_or_create_object(CLASSES[probas_sub[idx].argmax()], sub_bboxes_scaled[idx_in_kept])
        obj_idx:int = get_or_create_object(CLASSES[probas_obj[idx].argmax()], obj_bboxes_scaled[idx_in_kept])
        predicate = REL_CLASSES[probas[idx].argmax()]
        
        # Use tuple as key to prevent duplicate triplets
        triplet_key = (sub_idx, predicate, obj_idx)
        if triplet_key not in relationships:
            relationships[triplet_key] = {
                "predicate": predicate,
                "subject": sub_idx,
                "object": obj_idx
            }

    # Convert relationships dict to list for JSON output
    relationships_list = list(relationships.values())
    
    return {"url": image, "objects": objects, "attributes": attributes, "relationships": relationships_list}


def save_scene_graph_json(scene_graph, filepath):
    with open(filepath, 'w') as f:
        json.dump(scene_graph, f, indent=2)
    print(f"Scene graph saved to: {filepath}")


def print_scene_graph(scene_graph):
    print(f"\n{'='*60}")
    print(f"Scene Graph for: {scene_graph['url']}")
    print(f"{'='*60}")
    print(f"\nObjects ({len(scene_graph['objects'])} total):")
    for idx, obj in enumerate(scene_graph['objects']):
        print(f"  [{idx}] {obj['name']}")
    if scene_graph['attributes']:
        print(f"\nAttributes ({len(scene_graph['attributes'])} total):")
        for attr in scene_graph['attributes']:
            print(f"  {attr['attribute']} -> {scene_graph['objects'][attr['object']]['name']} [{attr['object']}]")
    print(f"\nRelationships ({len(scene_graph['relationships'])} total):")
    for rel in scene_graph['relationships']:
        print(f"  {scene_graph['objects'][rel['subject']]['name']} [{rel['subject']}] -> {rel['predicate']} -> {scene_graph['objects'][rel['object']]['name']} [{rel['object']}]")
    print(f"{'='*60}\n")


def visualize_old_style(model, args):
    im, img, probas, probas_sub, probas_obj, keep_queries, indices, sub_bboxes_scaled, obj_bboxes_scaled, keep = infer(model, args)

    conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []
    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
            lambda self, input, output: dec_attn_weights_sub.append(output[1])
        ),
        model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
            lambda self, input, output: dec_attn_weights_obj.append(output[1])
        )
    ]
    with torch.no_grad():
        model(img)
        for hook in hooks:
            hook.remove()

        conv_features = conv_features[0]
        dec_attn_weights_sub = dec_attn_weights_sub[0]
        dec_attn_weights_obj = dec_attn_weights_obj[0]

        h, w = conv_features['0'].tensors.shape[-2:]

        fig, axs = plt.subplots(ncols=len(indices), nrows=3, figsize=(22, 7))
        for idx, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                zip(keep_queries, axs.T, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
            ax = ax_i[0]
            ax.imshow(dec_attn_weights_sub[0, idx].view(h, w))
            ax.axis('off')
            ax.set_title(f'query id: {idx.item()}')
            ax = ax_i[1]
            ax.imshow(dec_attn_weights_obj[0, idx].view(h, w))
            ax.axis('off')
            ax = ax_i[2]
            ax.imshow(im)
            ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                       fill=False, color='blue', linewidth=2.5))
            ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                       fill=False, color='orange', linewidth=2.5))
            ax.axis('off')
            ax.set_title(CLASSES[probas_sub[idx].argmax()] + ' ' + REL_CLASSES[probas[idx].argmax()] + ' ' + CLASSES[probas_obj[idx].argmax()], fontsize=10)

        fig.tight_layout()
        plt.show()

def compare_filenames(folder1, folder2):
    """
    Compare filenames (without extensions) in two folders.
    
    Args:
        folder1: Path to first folder
        folder2: Path to second folder
    
    Returns:
        dict with keys:
            - 'only_in_folder1': list of files unique to folder1
            - 'only_in_folder2': list of files unique to folder2
            - 'not_shared': combined list of all non-shared files
    """
    # Get all files (not directories) from both folders
    files1 = {Path(f).stem: f for f in os.listdir(folder1) 
              if os.path.isfile(os.path.join(folder1, f))}
    files2 = {Path(f).stem: f for f in os.listdir(folder2) 
              if os.path.isfile(os.path.join(folder2, f))}
    
    # Find files only in folder1 and only in folder2
    only_in_folder1 = [files1[name] for name in files1.keys() - files2.keys()]
    only_in_folder2 = [files2[name] for name in files2.keys() - files1.keys()]
    
    return {
        'only_in_folder1': only_in_folder1,
        'only_in_folder2': only_in_folder2,
        'not_shared': only_in_folder1 + only_in_folder2
    }


def run(args):
    if not args.visualize:
        os.makedirs(args.output_folder, exist_ok=True)

    if args.img_path is None and args.img_folder is None:
        print("Error: Please provide either --img_path, --img_folder, or --compare_folders")
        exit(1)

    model = load_model(args)

    if args.img_path: # Process a single image
        if args.visualize:
            visualize_old_style(model, args)
        else:
            scene_graph = main(model, args,args.img_path)
            print_scene_graph(scene_graph)

            base_name = os.path.splitext(os.path.basename(args.img_path))[0]
            base_name = os.path.join(args.output_folder, base_name)
            save_scene_graph_json(scene_graph, f"{base_name}_scene_graph.json")

    if args.img_folder: # Process all images in the specified folder
        file_list = collect_images(args.img_folder, args.depth)
        # file_list = os.listdir(args.img_folder)
        if args.ignore_files:
            print(f"Comparing files in '{args.img_folder}' with '{args.output_folder}' to ignore shared files...")
            comparison = compare_filenames(args.img_folder, args.output_folder)
            print(f"Found {len(comparison['only_in_folder1'])} unique files in '{args.img_folder}' and {len(comparison['only_in_folder2'])} unique files in '{args.output_folder}'.")
            print(f"Processing {len(comparison['only_in_folder1'])} non-shared files from '{args.img_folder}'...")
            file_list = comparison['only_in_folder1']  # Only process files that are unique to img_folder
        for filename in tqdm(file_list):
            try:
                if not filename.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                    continue
                if args.ignore_files:
                    # If ignoring shared files, we need to check if this file is shared with the output folder
                    output_files = set(os.listdir(args.output_folder))
                    if filename in output_files:
                        continue
                img_path = os.path.join(args.img_folder, filename)
                print(f"\nProcessing file: {img_path}")
                
                scene_graph = main(model, args,img_path)
                print_scene_graph(scene_graph)

                base_name = os.path.join(args.output_folder, os.path.splitext(filename)[0])
                if args.test:
                    print(f"Test mode enabled, not saving JSON for {filename}")
                    return
                save_scene_graph_json(scene_graph, f"{base_name}.json")
            except Exception as e:
                print(f"Error processing {filename}: {e}, skipping this file.")
                save_scene_graph_json({"url": img_path, "error": str(e)}, f"{os.path.join(args.output_folder, os.path.splitext(filename)[0])}.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    parser.add_argument('--output_folder', type=str, default="",
                        help="Path to save scene graph as JSON")
    parser.add_argument('--visualize', action='store_true',
                        help="Show old-style visualization instead of scene graph")
    parser.add_argument('--img_folder',type=str, default=None,
                        help="Path to a folder containing multiple images to process (overrides --img_path)")
    parser.add_argument('--depth', type=int, default=1,
                        help="When comparing folders, how many levels of subdirectories to traverse (default: 1, meaning only the specified folder and not its subfolders)")
    parser.add_argument('--ignore_files', action='store_true', default=False,
                        help="When comparing folders, ignore files that are shared between them")
    parser.add_argument("--test", action='store_true', help="Run a quick test with a single image to verify setup")
    
    args = parser.parse_args()
    
    if args.test:
        profiler = cProfile.Profile()
        profiler.enable()

    run(args)
    if args.test:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(30)