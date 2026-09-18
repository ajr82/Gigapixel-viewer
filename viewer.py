import sys
import os
import json
import traceback
import mimetypes
from pathlib import Path

try:
    # Use ThreadingHTTPServer to load tiles simultaneously for massive speed improvements
    from http.server import ThreadingHTTPServer as ServerBase
except ImportError:
    from http.server import HTTPServer as ServerBase
from http.server import BaseHTTPRequestHandler

import threading
import urllib.parse
from PIL import Image, ImageDraw
import io
import base64

import numpy as np
try:
    import tifffile
    import zarr
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("Warning: tifffile or zarr not installed. OME-TIFF fallback will be disabled.")
    print("To enable OME-TIFF support: pip install tifffile zarr imagecodecs")

# Checking for required libraries
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QListWidget, QListWidgetItem, QLabel, QSplitter, QFileDialog,
        QToolBar, QStatusBar, QMessageBox, QProgressDialog, QDialog,
        QFormLayout, QLineEdit, QPushButton, QComboBox, QSlider
    )
    from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QObject, QThread
    from PyQt6.QtGui import QIcon, QPixmap, QImage, QColor, QFont, QPainter
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import QWebEnginePage
except ImportError:
    print("Error: PyQt6 or PyQt6-WebEngine is not installed.")
    print("Please install them using: pip install PyQt6 PyQt6-WebEngine")
    sys.exit(1)

# --- OPENSLIDE BINARY SETUP FOR WINDOWS ---
if hasattr(os, 'add_dll_directory'):
    openslide_bin_path = r"C:\openslide-win64\bin" 
    env_path = os.environ.get('OPENSLIDE_PATH')
    
    try:
        if env_path and os.path.exists(env_path):
            os.add_dll_directory(env_path)
        elif os.path.exists(openslide_bin_path):
            os.add_dll_directory(openslide_bin_path)
    except Exception as e:
        print(f"Warning: Failed to add DLL directory: {e}")
# ------------------------------------------

try:
    import openslide
except ImportError as e:
    print(f"Error: OpenSlide is not installed or could not be loaded ({e}).")
    print("Please install it using: pip install openslide-python")
    print("You also need the OpenSlide C library installed on your system.")
    print("See: https://openslide.org/download/")
    sys.exit(1)


# ---------------------------------------------------------
# 1. UI Additions: Image Settings Dialog
# ---------------------------------------------------------
class ImageSettingsDialog(QDialog):
    settings_changed = pyqtSignal(str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Settings")
        self.setFixedSize(350, 320)
        self.layout = QVBoxLayout(self)

        self.sliders = {}
        controls = [
            ("Brightness", 100, 0, 200, 100.0),
            ("Contrast", 100, 0, 200, 100.0),
            ("Gamma", 100, 10, 300, 100.0),
            ("Saturation", 100, 0, 200, 100.0),
            ("Hue", 0, 0, 360, 1.0)
        ]

        for name, default_val, min_val, max_val, div in controls:
            lbl = QLabel()
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(default_val)
            
            slider.valueChanged.connect(lambda v, n=name, l=lbl, d=div: self.update_label(l, n, v, d))
            slider.valueChanged.connect(self.emit_filters)
            
            self.layout.addWidget(lbl)
            self.layout.addWidget(slider)
            self.sliders[name] = slider
            self.update_label(lbl, name, default_val, div)

        self.layout.addSpacing(10)
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setStyleSheet("padding: 8px; font-weight: bold; background-color: #e2e8f0; border-radius: 4px;")
        reset_btn.clicked.connect(self.reset_sliders)
        self.layout.addWidget(reset_btn)

    def update_label(self, label, name, val, div):
        if name == "Hue":
            label.setText(f"<b>{name}</b>: {val}°")
        else:
            label.setText(f"<b>{name}</b>: {val/div:.2f}")

    def reset_sliders(self):
        self.sliders["Brightness"].setValue(100)
        self.sliders["Contrast"].setValue(100)
        self.sliders["Gamma"].setValue(100)
        self.sliders["Saturation"].setValue(100)
        self.sliders["Hue"].setValue(0)

    def emit_filters(self):
        b = self.sliders["Brightness"].value() / 100.0
        c = self.sliders["Contrast"].value() / 100.0
        g = self.sliders["Gamma"].value() / 100.0
        s = self.sliders["Saturation"].value() / 100.0
        h = self.sliders["Hue"].value()
        
        css_str = f"brightness({b}) contrast({c}) saturate({s}) hue-rotate({h}deg)"
        self.settings_changed.emit(css_str, g)


# ---------------------------------------------------------
# 2. Slide Backends (Handling different formats)
# ---------------------------------------------------------
class TiffSlideFallback:
    def __init__(self, filepath):
        self.filepath = filepath
        self.tif = tifffile.TiffFile(filepath)
        self.series = self.tif.series[0]
        self.axes = self.series.axes
        
        # Extract associated macro/label images if present in secondary OME-TIFF series
        self.associated_images = {}
        for i in range(1, len(self.tif.series)):
            try:
                s = self.tif.series[i]
                name = s.name.lower() if s.name else ""
                if "label" in name or "macro" in name or "barcode" in name:
                    arr = s.asarray()
                    if arr.ndim >= 2:
                        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                            self.associated_images['label'] = Image.fromarray(arr)
                        else:
                            self.associated_images['label'] = Image.fromarray(arr).convert('RGB')
            except Exception:
                pass
        
        self.zarr_store = self.series.aszarr()
        self.zarr_group = zarr.open(self.zarr_store, mode='r')
        
        if hasattr(self.zarr_group, 'keys'):
            arrays = []
            for k in self.zarr_group.keys():
                arrays.append((k, self.zarr_group[k]))
            arrays.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 0)
            self.pyramid = [a[1] for a in arrays]
        else:
            self.pyramid = [self.zarr_group]
            
        self.x_idx = self.axes.index('X') if 'X' in self.axes else -1
        self.y_idx = self.axes.index('Y') if 'Y' in self.axes else -2
        
        self.width = self.pyramid[0].shape[self.x_idx]
        self.height = self.pyramid[0].shape[self.y_idx]
        self.dimensions = (self.width, self.height)
        
        self.level_downsamples = []
        for arr in self.pyramid:
            lw = arr.shape[self.x_idx]
            self.level_downsamples.append(self.width / lw)
            
    def get_best_level_for_downsample(self, downsample):
        best_level = 0
        for i, ds in enumerate(self.level_downsamples):
            if ds <= downsample * 1.1:
                best_level = i
        return best_level
        
    def read_region(self, location, level, size):
        x0, y0 = location
        w, h = size
        
        ds = self.level_downsamples[level]
        xl = int(x0 / ds)
        yl = int(y0 / ds)
        
        arr = self.pyramid[level]
        
        w_safe = min(w, arr.shape[self.x_idx] - xl)
        h_safe = min(h, arr.shape[self.y_idx] - yl)
        if w_safe <= 0 or h_safe <= 0:
            return Image.new('RGB', size, (255, 255, 255))
            
        # Memory Safety Stride for Navigator Requests
        step = 1
        max_dim = 2048
        if w_safe > max_dim or h_safe > max_dim:
            step = max(int(w_safe / 1024), int(h_safe / 1024), 1)
            
        slices = [slice(None)] * arr.ndim
        slices[self.y_idx] = slice(yl, yl + h_safe, step)
        slices[self.x_idx] = slice(xl, xl + w_safe, step)
        
        if 'Z' in self.axes: slices[self.axes.index('Z')] = 0
        if 'T' in self.axes: slices[self.axes.index('T')] = 0
        
        data = arr[tuple(slices)]
        kept_axes = [self.axes[i] for i, s in enumerate(slices) if isinstance(s, slice) and i < len(self.axes)]
        
        if 'C' in kept_axes:
            c_now = kept_axes.index('C')
            data = np.moveaxis(data, c_now, -1)
            
        if data.dtype != np.uint8:
            if data.dtype == bool:
                data = data.astype(np.uint8) * 255
            else:
                data = data.astype(float)
                dmax = data.max()
                if dmax > 0: data = (data / dmax) * 255.0
                data = data.astype(np.uint8)
                
        while data.ndim > 3 and data.shape[0] == 1:
            data = data[0]
            
        if data.ndim == 2:
            data = np.stack((data, data, data), axis=-1)
        elif data.ndim >= 3:
            if data.shape[-1] == 1:
                data = np.stack((data[..., 0], data[..., 0], data[..., 0]), axis=-1)
            elif data.shape[-1] == 2:
                z_pad = np.zeros(data.shape[:-1] + (1,), dtype=np.uint8)
                data = np.concatenate((data, z_pad), axis=-1)
            elif data.shape[-1] > 3:
                data = data[..., :3]
                
        img = Image.fromarray(data)
        
        target_w = max(int(w / step), 1)
        target_h = max(int(h / step), 1)
        
        if img.size[0] != target_w or img.size[1] != target_h:
            bg = Image.new('RGB', (target_w, target_h), (255, 255, 255))
            bg.paste(img, (0, 0))
            return bg
            
        return img
        
    def close(self):
        self.tif.close()


class TileServer(ServerBase):
    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        self.slide = None
        self.slide_path = None
        self.osr_config = {}
        
        self.slide_lock = threading.Lock()
        self.active_readers = 0
        self.close_condition = threading.Condition(self.slide_lock)

    def set_slide(self, slide_path):
        with self.slide_lock:
            old_slide = self.slide
            self.slide = None 
            
            while self.active_readers > 0:
                self.close_condition.wait()
                
            if old_slide is not None:
                if hasattr(old_slide, 'close'):
                    try:
                        old_slide.close()
                    except:
                        pass
        
        new_slide = None
        try:
            if slide_path.lower().endswith('.ome.tif') or slide_path.lower().endswith('.ome.tiff'):
                if HAS_TIFFFILE:
                    raise ValueError("Bypassing OpenSlide for OME-TIFF formats.")
            new_slide = openslide.OpenSlide(slide_path)
        except Exception as e_os:
            if HAS_TIFFFILE:
                try:
                    new_slide = TiffSlideFallback(slide_path)
                except Exception as e_tif:
                    print(f"Fallback failed: {e_tif}")
                    return False
            else:
                print(f"OpenSlide rejected file and fallback unavailable: {e_os}")
                return False
                
        try:
            width, height = new_slide.dimensions
            self.tile_size = 256
            self.overlap = 0 
            
            new_config = {
                "Image": {
                    "xmlns": "http://schemas.microsoft.com/deepzoom/2008",
                    "Format": "jpeg",
                    "Overlap": str(self.overlap),
                    "TileSize": str(self.tile_size),
                    "Size": {
                        "Width": str(width),
                        "Height": str(height)
                    }
                }
            }
            
            with self.slide_lock:
                self.slide = new_slide
                self.slide_path = slide_path
                self.osr_config = new_config
            return True
        except Exception as e:
            print(f"Error opening slide: {e}")
            if new_slide and hasattr(new_slide, 'close'):
                try:
                    new_slide.close()
                except:
                    pass
            return False

class TileRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            parsed_path = urllib.parse.urlparse(self.path)
            path = parsed_path.path
            
            if path == '/' or path == '/index.html':
                self.serve_html()
            elif path == '/slide.dzi':
                self.serve_dzi()
            elif path == '/label.jpg':
                self.serve_label()
            elif path.startswith('/slide_files/'):
                self.serve_tile()
            else:
                self.send_error(404, "File not found")
        except Exception as e:
            print(f"Error handling request {self.path}: {e}")
            traceback.print_exc()
            try:
                self.send_error(500, f"Internal Server Error: {str(e)}")
            except:
                pass
                
    def log_message(self, format, *args):
        pass

    def serve_html(self):
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Gigapixel Viewer</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"></script>
            <style>
                body { margin: 0; padding: 0; background-color: #f8fafc; overflow: hidden; }
                #openseadragon1 { width: 100vw; height: 100vh; }
                .overlay-text { 
                    position: absolute; bottom: 10px; right: 10px; 
                    background: rgba(255,255,255,0.85); padding: 6px 10px; 
                    border-radius: 6px; font-family: sans-serif; font-size: 13px; z-index: 100;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #e2e8f0;
                }
                #scalebar-container {
                    display: none; position: absolute; bottom: 20px; left: 20px; 
                    z-index: 100; background: rgba(255,255,255,0.75); padding: 4px 10px; 
                    border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.2);
                }
                #scalebar-line { border-bottom: 3px solid #1e293b; width: 100px; margin: 0 auto; }
                #scalebar-text { text-align: center; font-size: 12px; font-family: sans-serif; font-weight: bold; color: #1e293b; margin-top: 4px; }
                
                #label-container {
                    display: none; position: absolute; top: 20px; left: 20px; 
                    z-index: 100; border: 2px solid #1e293b; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                    border-radius: 4px; background: white; max-width: 250px; max-height: 250px;
                }
                #label-container img { max-width: 100%; max-height: 100%; display: block; }
            </style>
        </head>
        <body>
            <svg style="width:0;height:0;position:absolute;">
                <filter id="gamma-filter">
                    <feComponentTransfer>
                        <feFuncR type="gamma" id="fe-r" amplitude="1" exponent="1" offset="0"/>
                        <feFuncG type="gamma" id="fe-g" amplitude="1" exponent="1" offset="0"/>
                        <feFuncB type="gamma" id="fe-b" amplitude="1" exponent="1" offset="0"/>
                    </feComponentTransfer>
                </filter>
            </svg>

            <div id="openseadragon1"></div>
            
            <div id="scalebar-container">
                <div id="scalebar-line"></div>
                <div id="scalebar-text">100 &mu;m</div>
            </div>
            
            <div id="label-container">
                <img id="label-img" src="" alt="Slide Label">
            </div>
            
            <div id="status" class="overlay-text">Ready</div>

            <script type="text/javascript">
                var viewer;
                var navActive = true;
                var scalebarActive = false;
                var labelActive = false;
                var defaultMpp = 0.25; 

                function initViewer(loadSource) {
                    var ts = new Date().getTime();
                    if (viewer) { viewer.destroy(); }
                    
                    var options = {
                        id: "openseadragon1",
                        prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
                        showNavigator: navActive,
                        navigatorPosition: "TOP_RIGHT",
                        animationTime: 0.5,
                        blendTime: 0.1,
                        constrainDuringPan: true,
                        maxZoomPixelRatio: 2,
                        minZoomImageRatio: 0.8
                    };
                    
                    if (loadSource) { options.tileSources = "/slide.dzi?t=" + ts; }
                    
                    viewer = OpenSeadragon(options);
                    
                    viewer.addHandler('open', function() {
                        document.getElementById('status').innerText = 'Image Loaded';
                        updateScalebar();
                        if (labelActive) {
                            document.getElementById('label-img').src = '/label.jpg?t=' + new Date().getTime();
                        }
                    });
                    
                    viewer.addHandler('open-failed', function() {
                        document.getElementById('status').innerText = 'Error loading image. Check server log.';
                        document.getElementById('status').style.color = 'red';
                    });

                    viewer.addHandler('animation', updateScalebar);
                }
                
                function updateScalebar() {
                    var container = document.getElementById('scalebar-container');
                    if (!scalebarActive || !viewer.viewport) {
                        container.style.display = 'none';
                        return;
                    }
                    container.style.display = 'block';

                    var line = document.getElementById('scalebar-line');
                    var text = document.getElementById('scalebar-text');
                    
                    var zoom = viewer.viewport.getZoom(true);
                    var imageZoom = viewer.viewport.viewportToImageZoom(zoom);
                    var currentMpp = defaultMpp / imageZoom;
                    
                    var desiredMicrons = 150 * currentMpp;
                    var niceMicrons = 1;
                    if (desiredMicrons > 1000) niceMicrons = Math.round(desiredMicrons/1000) * 1000;
                    else if (desiredMicrons > 100) niceMicrons = Math.round(desiredMicrons/100) * 100;
                    else if (desiredMicrons > 10) niceMicrons = Math.round(desiredMicrons/10) * 10;
                    else niceMicrons = Math.round(desiredMicrons);
                    
                    var actualPixels = niceMicrons / currentMpp;
                    line.style.width = actualPixels + 'px';
                    
                    if (niceMicrons >= 1000) {
                        text.innerHTML = (niceMicrons/1000) + ' mm';
                    } else {
                        text.innerHTML = niceMicrons + ' &mu;m';
                    }
                }
                
                window.onload = function() { initViewer(false); };
                
                window.addEventListener('message', function(event) {
                    if (event.data === 'reload') {
                        document.getElementById('status').innerText = 'Loading...';
                        document.getElementById('status').style.color = 'black';
                        initViewer(true);
                    } else if (typeof event.data === 'string' && event.data.startsWith('zoom:')) {
                        var mag = parseFloat(event.data.split(':')[1]);
                        if (viewer && viewer.viewport) {
                            var targetRatio = mag / 40.0;
                            var vpZoom = viewer.viewport.imageToViewportZoom(targetRatio);
                            viewer.viewport.zoomTo(vpZoom);
                        }
                    } else if (typeof event.data === 'string' && event.data.startsWith('filter:')) {
                        var parts = event.data.substring(7).split('|');
                        var gamma = parseFloat(parts[0]);
                        var cssFilters = parts[1];
                        
                        viewer.drawer.canvas.style.filter = cssFilters + ' url(#gamma-filter)';
                        
                        var svgVal = 1.0 / gamma;
                        document.getElementById('fe-r').setAttribute('exponent', svgVal);
                        document.getElementById('fe-g').setAttribute('exponent', svgVal);
                        document.getElementById('fe-b').setAttribute('exponent', svgVal);
                    } else if (event.data === 'toggle_navigator') {
                        navActive = !navActive;
                        if (viewer.navigator) {
                            viewer.navigator.element.style.display = navActive ? 'block' : 'none';
                        }
                    } else if (event.data === 'toggle_scalebar') {
                        scalebarActive = !scalebarActive;
                        updateScalebar();
                    } else if (event.data === 'toggle_label') {
                        labelActive = !labelActive;
                        var container = document.getElementById('label-container');
                        if (labelActive) {
                            document.getElementById('label-img').src = '/label.jpg?t=' + new Date().getTime();
                            container.style.display = 'block';
                        } else {
                            container.style.display = 'none';
                        }
                    }
                });
            </script>
        </body>
        </html>
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def serve_dzi(self):
        if not self.server.osr_config:
            self.send_error(404, "No slide loaded")
            return
            
        import xml.etree.ElementTree as ET
        root = ET.Element("Image", 
                          xmlns=self.server.osr_config["Image"]["xmlns"],
                          TileSize=self.server.osr_config["Image"]["TileSize"],
                          Overlap=self.server.osr_config["Image"]["Overlap"],
                          Format=self.server.osr_config["Image"]["Format"])
        
        size = ET.SubElement(root, "Size",
                             Width=self.server.osr_config["Image"]["Size"]["Width"],
                             Height=self.server.osr_config["Image"]["Size"]["Height"])
                             
        xml_str = ET.tostring(root, encoding='utf8', method='xml')
        
        self.send_response(200)
        self.send_header('Content-type', 'application/xml')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.end_headers()
        self.wfile.write(xml_str)

    def serve_label(self):
        with self.server.slide_lock:
            slide = self.server.slide
            if slide:
                self.server.active_readers += 1
                
        if not slide:
            self.send_error(404, "No slide loaded")
            return

        try:
            img = None
            if hasattr(slide, 'associated_images'):
                if 'label' in slide.associated_images:
                    img = slide.associated_images['label']
                elif 'macro' in slide.associated_images:
                    img = slide.associated_images['macro']
            
            # Fallback if no label is found
            if not img:
                img = Image.new('RGB', (200, 100), (240, 240, 240))
                d = ImageDraw.Draw(img)
                d.text((50, 40), "No Label Found", fill=(100, 100, 100))

            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            img_io = io.BytesIO()
            img.save(img_io, 'JPEG', quality=85)
            img_io.seek(0)
            
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(img_io.getvalue())
        except Exception as e:
            print(f"Error serving label: {e}")
            self.send_error(500, "Error generating label")
        finally:
            with self.server.slide_lock:
                self.server.active_readers -= 1
                if self.server.active_readers == 0:
                    self.server.close_condition.notify_all()

    def serve_tile(self):
        parts = self.path.split('/')
        if len(parts) < 4:
            self.send_error(400, "Invalid tile request")
            return
            
        try:
            level = int(parts[2])
            filename = parts[3].split('?')[0] 
            col_str, rest = filename.split('_')
            row_str, format = rest.split('.')
            col = int(col_str)
            row = int(row_str)
        except ValueError:
            self.send_error(400, "Invalid tile coordinates")
            return
            
        with self.server.slide_lock:
            slide = self.server.slide
            if slide:
                self.server.active_readers += 1
                
        if not slide:
            self.send_error(404, "No slide loaded")
            return

        try:
            import math
            width, height = self.server.slide.dimensions
            max_dim = max(width, height)
            
            if max_dim <= 0:
                self.send_error(400, "Invalid image dimensions")
                return
                
            max_dzi_level = int(math.ceil(math.log(max_dim, 2)))
            downsample = math.pow(2, max_dzi_level - level)
            
            best_os_level = self.server.slide.get_best_level_for_downsample(downsample)
            os_level_downsample = self.server.slide.level_downsamples[best_os_level]
            
            tile_size = self.server.tile_size
            
            x_level = col * tile_size
            y_level = row * tile_size
            
            x0 = int(x_level * downsample)
            y0 = int(y_level * downsample)
            
            level_width = int(math.ceil(width / downsample))
            level_height = int(math.ceil(height / downsample))
            
            w_level = min(tile_size, level_width - x_level)
            h_level = min(tile_size, level_height - y_level)
            
            w0 = int(w_level * downsample)
            h0 = int(h_level * downsample)
            
            if w0 <= 0 or h0 <= 0:
                img = Image.new('RGB', (tile_size, tile_size), color=(255, 255, 255))
            else:
                w_os = int(w0 / os_level_downsample)
                h_os = int(h0 / os_level_downsample)
                
                w_os = max(1, w_os)
                h_os = max(1, h_os)
                
                region = self.server.slide.read_region((x0, y0), best_os_level, (w_os, h_os))
                
                if region.mode == 'RGBA':
                    bg = Image.new('RGB', region.size, (255, 255, 255))
                    bg.paste(region, mask=region.split()[3])
                    img = bg
                else:
                    img = region.convert('RGB')
                
            if (img.size[0] != w_level) or (img.size[1] != h_level):
                resample_filter = Image.LANCZOS if (img.size[0] > w_level) else Image.BICUBIC
                img = img.resize((w_level, h_level), resample=resample_filter)
        
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            img_io = io.BytesIO()
            img.save(img_io, 'JPEG', quality=85)
            img_io.seek(0)
            
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.send_header('Content-Length', str(len(img_io.getvalue())))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'public, max-age=31536000')
            self.end_headers()
            
            self.wfile.write(img_io.getvalue())
            
        except Exception as e:
            import traceback
            print(f"Error serving tile {level}/{col}_{row}: {e}")
            traceback.print_exc()
            self.send_error(500, "Error generating tile")
        finally:
            with self.server.slide_lock:
                self.server.active_readers -= 1
                if self.server.active_readers == 0:
                    self.server.close_condition.notify_all()

class ServerThread(QThread):
    def __init__(self, port=8000):
        super().__init__()
        self.port = port
        self.server = None
        
    def run(self):
        max_attempts = 10
        for p in range(self.port, self.port + max_attempts):
            try:
                self.server = TileServer(('127.0.0.1', p), TileRequestHandler)
                self.port = p
                print(f"Started internal tile server on port {self.port}")
                self.server.serve_forever()
                break
            except OSError as e:
                if e.errno == 98 or e.errno == 10048: 
                    print(f"Port {p} in use, trying next...")
                    continue
                else:
                    print(f"Error starting server: {e}")
                    break
                    
    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    def load_slide(self, path):
        if self.server:
            return self.server.set_slide(path)
        return False

class SlideOpenerThread(QThread):
    finished = pyqtSignal(bool, str)
    
    def __init__(self, server_thread, filepath):
        super().__init__()
        self.server_thread = server_thread
        self.filepath = filepath
        
    def run(self):
        success = False
        try:
            success = self.server_thread.load_slide(self.filepath)
        except Exception as e:
            print(f"Error opening slide in background: {e}")
        self.finished.emit(success, self.filepath)


# ---------------------------------------------------------
# 3. GUI Layout & Actions Integration
# ---------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 1. NEW: Enable drag and drop functionality on the main window
        self.setAcceptDrops(True)
        
        self.setWindowTitle("Gigapixel viewer")
        self.resize(1280, 800)
        self.setMinimumSize(800, 600)
        
        icon_pixmap = QPixmap(64, 64)
        icon_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(icon_pixmap)
        font = QFont()
        font.setPixelSize(48)
        painter.setFont(font)
        painter.drawText(icon_pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "🔬")
        painter.end()
        self.setWindowIcon(QIcon(icon_pixmap))
        
        self.supported_exts = ['.svs', '.vms', '.vmu', '.ndpi', '.scn', '.mrxs', '.tiff', '.tif', '.bif', '.ome.tif', '.ome.tiff']
        self.current_directory = None
        
        self.server_thread = ServerThread(port=8080)
        self.server_thread.start()
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        open_dir_action = toolbar.addAction("📂 Open Directory")
        open_dir_action.triggered.connect(self.open_directory)
        
        open_file_action = toolbar.addAction("📄 Open File")
        open_file_action.triggered.connect(self.open_single_file)
        
        toolbar.addSeparator()
        
        for mag in [2, 4, 10, 20, 40]:
            btn = toolbar.addAction(f"{mag}x")
            btn.triggered.connect(lambda checked=False, m=mag: self.set_magnification(m))
        
        toolbar.addSeparator()

        btn_settings = toolbar.addAction("🎛️ Image")
        btn_settings.triggered.connect(self.open_settings)

        btn_scalebar = toolbar.addAction("📏 Scalebar")
        btn_scalebar.triggered.connect(self.toggle_scalebar)

        btn_navigator = toolbar.addAction("🗺️ Navigator")
        btn_navigator.triggered.connect(self.toggle_navigator)
        
        btn_label = toolbar.addAction("🏷️ Label")
        btn_label.triggered.connect(self.toggle_label)
        
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(5)
        
        self.dir_label = QLabel("No directory selected")
        self.dir_label.setStyleSheet("padding: 8px; background-color: #f1f5f9; border-radius: 4px; font-weight: bold; color: #334155;")
        left_layout.addWidget(self.dir_label)
        
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.file_list.itemClicked.connect(self.on_file_selected)
        left_layout.addWidget(self.file_list)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.viewer_web = QWebEngineView()
        right_layout.addWidget(self.viewer_web)
        
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(right_panel)
        
        self.splitter.setSizes([250, 1030])
        
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready. Select a directory or file to begin.")
        
        self.load_viewer_page()

    # --- NEW: Drag and Drop Methods ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                self.load_directory(path)
                return
            elif os.path.isfile(path):
                # Check for supported extensions, including compound extensions like .ome.tif
                ext = Path(path).suffix.lower()
                if path.lower().endswith('.ome.tif'):
                    ext = '.ome.tif'
                if ext in self.supported_exts:
                    self.add_single_file(path)
                    return
    # -----------------------------------

    def open_settings(self):
        if not hasattr(self, 'settings_dialog'):
            self.settings_dialog = ImageSettingsDialog(self)
            self.settings_dialog.settings_changed.connect(self.apply_image_settings)
        self.settings_dialog.show()
        self.settings_dialog.raise_()

    def apply_image_settings(self, css_str, gamma):
        js = f"window.postMessage('filter:{gamma}|{css_str}', '*');"
        self.viewer_web.page().runJavaScript(js)

    def toggle_scalebar(self):
        js = "window.postMessage('toggle_scalebar', '*');"
        self.viewer_web.page().runJavaScript(js)

    def toggle_navigator(self):
        js = "window.postMessage('toggle_navigator', '*');"
        self.viewer_web.page().runJavaScript(js)
        
    def toggle_label(self):
        js = "window.postMessage('toggle_label', '*');"
        self.viewer_web.page().runJavaScript(js)

    def open_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory with Pathology Images")
        if dir_path:
            self.load_directory(dir_path)
            
    def open_single_file(self):
        filter_str = "Pathology Images (*.svs *.ndpi *.tiff *.tif *.mrxs *.scn *.vms *.vmu);;All Files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Pathology Image", "", filter_str)
        if file_path:
            self.add_single_file(file_path)

    def load_directory(self, dir_path):
        self.current_directory = dir_path
        self.dir_label.setText(f"📁 Folder: {Path(dir_path).name}")
        self.file_list.clear()
        
        filepaths = []
        try:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    ext = Path(file).suffix.lower()
                    if file.lower().endswith('.ome.tif'):
                        ext = '.ome.tif'
                    if ext in self.supported_exts:
                        full_path = os.path.join(root, file)
                        filepaths.append(full_path)
                        
                        item = QListWidgetItem(" " + file)
                        item.setData(Qt.ItemDataRole.UserRole, full_path)
                        self.file_list.addItem(item)
                        
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to read directory:\n{e}")
            return
            
        self.statusBar.showMessage(f"Found {len(filepaths)} compatible images.")
            
    def add_single_file(self, file_path):
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == file_path:
                self.file_list.setCurrentItem(item)
                self.on_file_selected(item)
                return
                
        filename = Path(file_path).name
        item = QListWidgetItem(" " + filename)
        item.setData(Qt.ItemDataRole.UserRole, file_path)
        
        self.file_list.addItem(item)
        self.file_list.setCurrentItem(item)
        self.on_file_selected(item)

    def load_viewer_page(self):
        url = QUrl(f"http://127.0.0.1:{self.server_thread.port}/index.html")
        self.viewer_web.setUrl(url)

    def set_magnification(self, mag):
        js = f"window.postMessage('zoom:{mag}', '*');"
        self.viewer_web.page().runJavaScript(js)

    def on_file_selected(self, item):
        filepath = item.data(Qt.ItemDataRole.UserRole)
        self.expected_filepath = filepath
        self.statusBar.showMessage(f"Opening {Path(filepath).name}... Please wait.")
        
        js = "if (document.getElementById('status')) { document.getElementById('status').innerText = 'Opening image file...'; document.getElementById('status').style.color = 'black'; }"
        self.viewer_web.page().runJavaScript(js)
        
        self.opener_thread = SlideOpenerThread(self.server_thread, filepath)
        self.opener_thread.finished.connect(self.on_slide_opened)
        self.opener_thread.start()

    def on_slide_opened(self, success, filepath):
        if filepath != getattr(self, 'expected_filepath', None):
            return
            
        if success:
            js = "window.postMessage('reload', '*');"
            self.viewer_web.page().runJavaScript(js)
            self.statusBar.showMessage(f"Loaded {Path(filepath).name}")
        else:
            self.statusBar.showMessage(f"Failed to load {Path(filepath).name}")
            QMessageBox.warning(self, "Error", f"Failed to open image:\n{filepath}\n\nIt might be corrupted or unsupported by your OpenSlide installation.")

    def closeEvent(self, event):
        if self.server_thread:
            self.server_thread.stop()
            self.server_thread.wait()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    style = """
    QMainWindow {
        background-color: #f8fafc;
    }
    QListWidget {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        outline: none;
        font-size: 14px;
        color: #334155;
    }
    QListWidget::item {
        border-bottom: 1px solid #f1f5f9;
        padding: 10px 4px;
    }
    QListWidget::item:selected {
        background-color: #e0f2fe;
        color: #0284c7;
        font-weight: bold;
        border-radius: 4px;
    }
    QListWidget::item:hover:!selected {
        background-color: #f1f5f9;
        border-radius: 4px;
    }
    QToolBar {
        background-color: #ffffff;
        border-bottom: 1px solid #e2e8f0;
        padding: 8px;
        spacing: 10px;
    }
    QToolButton {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 13px;
        font-weight: 600;
        color: #475569;
    }
    QToolButton:hover {
        background-color: #e2e8f0;
        color: #0f172a;
    }
    QToolButton:pressed {
        background-color: #cbd5e1;
    }
    QSplitter::handle {
        background-color: #e2e8f0;
        width: 2px;
    }
    QStatusBar {
        background-color: #ffffff;
        color: #64748b;
        border-top: 1px solid #e2e8f0;
    }
    """
    app.setStyleSheet(style)
    
    window = MainWindow()
    window.show()

    # 2. NEW: "Open With..." CLI argument parsing
    # Automatically load a file or directory if it was passed via command line
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if os.path.exists(filepath):
            if os.path.isdir(filepath):
                window.load_directory(filepath)
            else:
                window.add_single_file(filepath)
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()