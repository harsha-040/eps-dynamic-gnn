#!/usr/bin/env python3
"""Helpers for generating paragraph XML matching the CMRIT template's native style."""
import html
import re

def esc(s):
    return html.escape(s, quote=False)

def bullet_p(text, lvl=0, bold=False, color=None, size=None):
    """Level-0/1 arrow-bulleted paragraph (► for lvl0, matches template)."""
    marL = 457200 if lvl == 0 else 900000
    indent = -320040 if lvl == 0 else -320040
    rpr_attrs = []
    if bold:
        rpr_attrs.append('b="1"')
    if size:
        rpr_attrs.append(f'sz="{size}"')
    rpr_attrs.append('lang="en-US"')
    rpr = f'<a:rPr {" ".join(rpr_attrs)}>' if (bold or size or color) else '<a:rPr lang="en-US"/>'
    fill = f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>' if color else ''
    if bold or size or color:
        rpr = f'<a:rPr {" ".join(rpr_attrs)}>{fill}</a:rPr>'
    return (f'<a:p><a:pPr indent="{indent}" lvl="{lvl}" marL="{marL}" rtl="0" algn="l">'
            f'<a:lnSpc><a:spcPct val="100000"/></a:lnSpc><a:spcBef><a:spcPts val="1000"/></a:spcBef>'
            f'<a:spcAft><a:spcPts val="0"/></a:spcAft><a:buSzPts val="1440"/><a:buChar char="&#9658;"/></a:pPr>'
            f'<a:r>{rpr}<a:t>{esc(text)}</a:t></a:r><a:endParaRPr/></a:p>')

def numbered_p(text, lvl=0, bold=False, color=None, size=None):
    marL = 342900 if lvl == 0 else 800100
    indent = -342900
    rpr_attrs = ['lang="en-US"']
    if bold:
        rpr_attrs.append('b="1"')
    if size:
        rpr_attrs.append(f'sz="{size}"')
    fill = f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>' if color else ''
    rpr = f'<a:rPr {" ".join(rpr_attrs)}>{fill}</a:rPr>'
    return (f'<a:p><a:pPr indent="{indent}" lvl="{lvl}" marL="{marL}" rtl="0" algn="l">'
            f'<a:lnSpc><a:spcPct val="100000"/></a:lnSpc><a:spcBef><a:spcPts val="1000"/></a:spcBef>'
            f'<a:spcAft><a:spcPts val="0"/></a:spcAft><a:buSzPts val="1440"/><a:buAutoNum type="arabicPeriod"/></a:pPr>'
            f'<a:r>{rpr}<a:t>{esc(text)}</a:t></a:r><a:endParaRPr/></a:p>')

def plain_p(text="", bold=False, italic=False, color=None, size=None, align="l"):
    rpr_attrs = ['lang="en-US"']
    if bold:
        rpr_attrs.append('b="1"')
    if italic:
        rpr_attrs.append('i="1"')
    if size:
        rpr_attrs.append(f'sz="{size}"')
    fill = f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>' if color else ''
    rpr = f'<a:rPr {" ".join(rpr_attrs)}>{fill}</a:rPr>' if len(rpr_attrs) > 1 or color else '<a:rPr lang="en-US"/>'
    if text:
        run = f'<a:r>{rpr}<a:t>{esc(text)}</a:t></a:r>'
    else:
        run = ''
    return (f'<a:p><a:pPr indent="0" lvl="0" marL="0" rtl="0" algn="{align}">'
            f'<a:lnSpc><a:spcPct val="100000"/></a:lnSpc><a:spcBef><a:spcPts val="1000"/></a:spcBef>'
            f'<a:spcAft><a:spcPts val="0"/></a:spcAft><a:buSzPts val="1440"/><a:buNone/></a:pPr>'
            f'{run}<a:endParaRPr/></a:p>')

def rich_bold_p(bold_prefix, rest, lvl=0):
    """A bulleted paragraph with a bold lead-in (e.g. 'Accuracy: ') then normal text."""
    marL = 457200
    indent = -320040
    return (f'<a:p><a:pPr indent="{indent}" lvl="{lvl}" marL="{marL}" rtl="0" algn="l">'
            f'<a:lnSpc><a:spcPct val="100000"/></a:lnSpc><a:spcBef><a:spcPts val="1000"/></a:spcBef>'
            f'<a:spcAft><a:spcPts val="0"/></a:spcAft><a:buSzPts val="1440"/><a:buChar char="&#9658;"/></a:pPr>'
            f'<a:r><a:rPr b="1" lang="en-US"/><a:t>{esc(bold_prefix)}</a:t></a:r>'
            f'<a:r><a:rPr lang="en-US"/><a:t>{esc(rest)}</a:t></a:r><a:endParaRPr/></a:p>')


def replace_title(xml, new_title):
    """Replace text inside the 'title' placeholder shape."""
    m = re.search(r'(<p:ph type="title"/>.*?<a:t>)([^<]*)(</a:t>)', xml, re.S)
    if not m:
        raise ValueError("title placeholder not found")
    return xml[:m.start(2)] + esc(new_title) + xml[m.end(2):]


def replace_body(xml, paragraphs_xml, ph_idx=1):
    """Replace all paragraphs inside the 'body' placeholder's <p:txBody> with new ones."""
    pat = re.compile(
        r'(<p:ph idx="' + str(ph_idx) + r'" type="body"/>.*?<p:txBody>(?:(?!</p:txBody>).)*?<a:lstStyle/>)(.*?)(</p:txBody>)',
        re.S)
    m = pat.search(xml)
    if not m:
        # try without idx constraint
        pat = re.compile(r'(<p:ph idx="1" type="body"/>.*?<a:lstStyle/>)(.*?)(</p:txBody>)', re.S)
        m = pat.search(xml)
    if not m:
        raise ValueError("body placeholder not found")
    return xml[:m.start(2)] + paragraphs_xml + xml[m.end(2):]


if __name__ == "__main__":
    print(bullet_p("test bullet"))
    print(numbered_p("test numbered", lvl=1))
