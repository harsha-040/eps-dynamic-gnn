#!/usr/bin/env python3
"""Helpers for inserting images into pptx slides (standard transitional OOXML)."""
import os
import re
import shutil
from PIL import Image

W = os.path.dirname(os.path.abspath(__file__))
UNPACKED = os.path.join(W, "unpacked")
MEDIA_DIR = os.path.join(UNPACKED, "ppt", "media")

def _current_max_image_num():
    if not os.path.isdir(MEDIA_DIR):
        return 0
    nums = []
    for fn in os.listdir(MEDIA_DIR):
        m = re.match(r"image(\d+)\.", fn)
        if m:
            nums.append(int(m.group(1)))
    return max(nums, default=0)

def _next_media_name(src_path):
    n = _current_max_image_num() + 1
    ext = os.path.splitext(src_path)[1].lstrip(".")
    return f"image{n}.{ext}"


def add_picture_to_slide(slide_num, img_path, x, y, cx=None, cy=None, name=None):
    """Copies img_path into ppt/media, registers a relationship for slideN,
    and returns the <p:pic>...</p:pic> XML to insert into the slide's spTree.
    x,y,cx,cy in EMU. If cx/cy omitted, cy is computed from cx and aspect ratio
    (cx must be given), or both must be given.
    """
    im = Image.open(img_path)
    ar = im.size[0] / im.size[1]
    if cx and not cy:
        cy = int(cx / ar)
    elif cy and not cx:
        cx = int(cy * ar)

    media_name = _next_media_name(img_path)
    shutil.copy(img_path, os.path.join(MEDIA_DIR, media_name))

    rels_path = os.path.join(UNPACKED, "ppt", "slides", "_rels", f"slide{slide_num}.xml.rels")
    with open(rels_path, encoding="utf-8") as f:
        rels_xml = f.read()
    existing_ids = re.findall(r'Id="rId(\d+)"', rels_xml)
    next_id = max([int(i) for i in existing_ids], default=0) + 1
    rid = f"rId{next_id}"
    new_rel = (f'<Relationship Id="{rid}" '
               f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
               f'Target="../media/{media_name}"/>')
    rels_xml = rels_xml.replace("</Relationships>", new_rel + "</Relationships>")
    with open(rels_path, "w", encoding="utf-8") as f:
        f.write(rels_xml)

    pic_id = 900 + next_id
    shape_name = name or f"Picture {pic_id}"
    pic_xml = (
        f'<p:pic><p:nvPicPr><p:cNvPr id="{pic_id}" name="{shape_name}"/>'
        f'<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
        f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
        f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
    )
    return pic_xml


def insert_before_close_sptree(xml, pic_xml):
    idx = xml.rfind("</p:spTree>")
    if idx == -1:
        raise ValueError("</p:spTree> not found")
    return xml[:idx] + pic_xml + xml[idx:]
