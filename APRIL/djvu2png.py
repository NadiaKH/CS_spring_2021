#!/usr/bin/env python
# encoding=UTF-8

# Copyright © 2010-2021 Jakub Wilk <jwilk@jwilk.net>
#
# This file is part of python-djvulibre.
#
# python-djvulibre is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License version 2 as published by
# the Free Software Foundation.
#
# python-djvulibre is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.

from __future__ import print_function

import os
import sys

import cairo
import djvu.decode
import numpy

cairo_pixel_format = cairo.FORMAT_ARGB32
djvu_pixel_format = djvu.decode.PixelFormatRgbMask(0xFF0000, 0xFF00, 0xFF, bpp=32)
djvu_pixel_format.rows_top_to_bottom = 1
djvu_pixel_format.y_top_to_bottom = 0

class Context(djvu.decode.Context):

    def handle_message(self, message):
        if isinstance(message, djvu.decode.ErrorMessage):
            print(message, file=sys.stderr)
            # Exceptions in handle_message() are ignored, so sys.exit()
            # wouldn't work here.
            os._exit(1)

    def process(self, djvu_path, png_folder_path, mode):
        document = self.new_document(djvu.decode.FileURI(djvu_path))
        document.decoding_job.wait()
        for i, page in enumerate(document.pages):
            print(i)
            page_job = page.decode(wait=True)
            width, height = page_job.size
            rect = (0, 0, width, height)
            bytes_per_line = cairo.ImageSurface.format_stride_for_width(cairo_pixel_format, width)
            assert bytes_per_line % 4 == 0
            color_buffer = numpy.zeros((height, bytes_per_line // 4), dtype=numpy.uint32)
            page_job.render(mode, rect, rect, djvu_pixel_format, row_alignment=bytes_per_line, buffer=color_buffer)
            mask_buffer = numpy.zeros((height, bytes_per_line // 4), dtype=numpy.uint32)
            if mode == djvu.decode.RENDER_FOREGROUND:
                page_job.render(djvu.decode.RENDER_MASK_ONLY, rect, rect, djvu_pixel_format, row_alignment=bytes_per_line, buffer=mask_buffer)
                mask_buffer <<= 24
                color_buffer |= mask_buffer
            color_buffer ^= 0xFF000000
            surface = cairo.ImageSurface.create_for_data(color_buffer, cairo_pixel_format, width, height)
            
            save_path = os.path.join(png_folder_path,  str(i) + '.png')
            surface.write_to_png(save_path)
            

def main():
    
    djvu_path = sys.argv[1]
    png_folder_path = sys.argv[2]
    
        
    
    context = Context()
    context.process(djvu_path, png_folder_path, djvu.decode.RENDER_COLOR)
    
    
if __name__ == '__main__':
    main()
    
    