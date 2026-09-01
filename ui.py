"""
ui.py - Sulu Boya Simulasyonu Arayuzu
Desteklenen Modlar:
1. Firca Modu (Surekli Surukleme / Continuous Brush Strokes)
2. Damla Modu (Tekil Tiklama / Discrete Drops & Splatters)
"""

import sys
import os
import time
import math
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSlider,
    QPushButton, QColorDialog, QListWidget, QListWidgetItem,
    QVBoxLayout, QHBoxLayout, QGroupBox, QTabWidget,
    QStatusBar, QSplitter, QScrollArea, QSizePolicy,
    QFileDialog, QMessageBox, QFrame, QRadioButton, QButtonGroup,
    QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect, QPointF
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QBrush,
    QFont, QCursor, QPalette, QPainterPath
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import WaterColorCanvas

CANVAS_W      = 1000
CANVAS_H      = 800
PREVIEW_SCALE = 0.4


class ParamRow(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, label, mn, mx, default, decimals=2):
        super().__init__()
        self._mn  = mn
        self._mx  = mx
        self._dec = decimals

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(6)

        lbl = QLabel(label)
        lbl.setFixedWidth(112)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(self._encode(default))

        self.val_lbl = QLabel(self._fmt(default))
        self.val_lbl.setFixedWidth(46)
        self.val_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        self.slider.valueChanged.connect(self._on_change)
        layout.addWidget(lbl)
        layout.addWidget(self.slider)
        layout.addWidget(self.val_lbl)

    def _encode(self, v):
        return int((v - self._mn) / (self._mx - self._mn) * 1000)

    def _decode(self, i):
        return self._mn + i / 1000 * (self._mx - self._mn)

    def _fmt(self, v):
        return f"{v:.{self._dec}f}"

    def _on_change(self, i):
        v = self._decode(i)
        self.val_lbl.setText(self._fmt(v))
        self.valueChanged.emit(v)

    def value(self):
        return self._decode(self.slider.value())

    def setValue(self, v):
        self.slider.setValue(self._encode(v))


class RenderWorker(QThread):
    finished  = pyqtSignal(object)
    statusMsg = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def run(self):
        t0 = time.time()
        try:
            cfg   = self.cfg
            scale = PREVIEW_SCALE if cfg["preview"] else 1.0
            w, h  = int(CANVAS_W * scale), int(CANVAS_H * scale)

            self.statusMsg.emit("Kagit dokusu olusturuluyor...")
            wc = WaterColorCanvas(width=w, height=h)
            pp = cfg["paper"]
            wc.generate_paper_texture(
                scale=pp["scale"],
                octaves=pp["octaves"],
                intensity=pp["intensity"],
            )

            # 1. FIRÇA DARBELERİ (STROKES)
            strokes = cfg.get("strokes", [])
            if strokes:
                # Tüm darbeleri TEK BİR BİRLEŞİK AKIŞ (Unified Wash) olarak işle!
                # Renkler farklı olsa dahi (Graded Wash), manifold tüm renkleri Shepard enterpolasyonuyla
                # pürüzsüzce birbirine bağlar, beyaz çizgileri ve basamaklanmaları %100 yok eder!
                scaled_strokes = []
                for st in strokes:
                    scaled_pts = [(p[0] * scale, p[1] * scale) for p in st["points"]]
                    st_copy = dict(st)
                    st_copy["points"] = scaled_pts
                    st_copy["radius"] = max(3.0, st["radius"] * scale)
                    scaled_strokes.append(st_copy)

                self.statusMsg.emit(f"Birlesik firca akisi ({len(scaled_strokes)} darbe) isleniyor...")
                if len(scaled_strokes) > 1:
                    wc.add_strokes_wash(scaled_strokes)
                else:
                    st = scaled_strokes[0]
                    wc.add_stroke(
                        points=st["points"],
                        color=st["color"],
                        base_radius=st["radius"],
                        intensity=st["intensity"],
                        edge_darkening=st["edge_darkening"],
                        dry_brush=st["dry_brush"],
                        bristle_strength=st["bristle_strength"],
                        paper_interaction=st["paper_interaction"],
                    )

            # 2. BİRLEŞİK DAMLALAR (DROPS)
            drops = cfg.get("drops", [])
            if drops:
                self.statusMsg.emit("Birlesik damla alani isleniyor...")
                scaled_drops = []
                for d in drops:
                    x = max(0, min(w - 1, int(d["x"] * scale)))
                    y = max(0, min(h - 1, int(d["y"] * scale)))
                    r = max(5, int(d["radius"] * scale))
                    scaled_drops.append({
                        "x": x, "y": y, "color": d["color"],
                        "radius": r,
                        "intensity": d["intensity"],
                        "edge_darkening": d["edge_darkening"],
                        "feathering": d["feathering"],
                        "paper_interaction": d["paper_interaction"],
                    })

                wc.add_pigment_cluster(scaled_drops)

                for i, d in enumerate(drops):
                    x = max(0, min(w - 1, int(d["x"] * scale)))
                    y = max(0, min(h - 1, int(d["y"] * scale)))
                    r = max(5, int(d["radius"] * scale))

                    if d.get("bloom_enabled"):
                        wc.apply_wet_bloom(
                            center_x=x, center_y=y, radius=r,
                            strength=d["bloom_strength"],
                            bloom_size=d["bloom_size"],
                        )
                    if d.get("splatter_enabled"):
                        wc.add_splatter(
                            center_x=x, center_y=y, color=d["color"],
                            radius=r,
                            count=d["splatter_count"],
                            scatter_min=d["scatter_min"],
                            scatter_max=d["scatter_max"],
                            intensity=d["splatter_intensity"],
                            wet_react=d.get("splatter_wet_react", 0.85),
                            dispersion=d.get("splatter_dispersion", 1.70),
                        )

            # 3. AKIŞ VE DİFÜZYON
            fl = cfg["flow"]
            self.statusMsg.emit("Akis simulasyonu...")
            wc.simulate_flow(
                iterations=fl["iterations"],
                diffusion_rate=fl["diffusion_rate"],
                turbulence=fl["turbulence"],
                gravity_strength=fl["gravity_strength"],
                gravity_angle=fl["gravity_angle"],
            )

            # 4. KURUMA
            dr = cfg["drying"]
            self.statusMsg.emit("Kuruma efekti...")
            wc.apply_drying_shift(
                value_shift=dr["value_shift"],
                saturation_shift=dr["saturation_shift"],
            )

            elapsed = time.time() - t0
            self.statusMsg.emit(f"Tamamlandi - {elapsed:.1f}s")
            arr = np.clip(wc.canvas, 0, 255).astype(np.uint8)
            self.finished.emit((arr, cfg["preview"]))

        except Exception as exc:
            self.statusMsg.emit(f"Hata: {exc}")
            self.finished.emit(None)


class CanvasWidget(QLabel):
    dropPlaced      = pyqtSignal(float, float)
    strokeCompleted = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(500, 380)
        self.setStyleSheet("background:#1a1a1a; border:1px solid #444;")
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

        self.mode = "STROKE"  # "STROKE" veya "DROP"
        self.cur_brush_color = (40, 120, 200)
        self.cur_brush_radius = 25.0
        self.show_guides = True
        self.selected_stroke_idx = -1
        self.selected_marker_idx = -1

        self._base_pm = None
        self._markers = []
        self._strokes_paths = []
        self._current_stroke = []
        self._is_drawing = False

        self._init_placeholder()

    def _init_placeholder(self):
        pm = QPixmap(CANVAS_W, CANVAS_H)
        pm.fill(QColor(240, 240, 226))
        p = QPainter(pm)
        p.setPen(QColor(160, 150, 140))
        p.setFont(QFont("Segoe UI", 20))
        p.drawText(
            pm.rect(), Qt.AlignmentFlag.AlignCenter,
            "Firca ile cizin veya Damla ekleyin\nArdindan 'Onizleme'ye basin",
        )
        p.end()
        self._base_pm = pm
        self._redraw()

    def setImage(self, arr):
        h, w, _ = arr.shape
        qimg = QImage(arr.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        self._base_pm = QPixmap.fromImage(qimg)
        self._redraw()

    def addMarker(self, norm_x, norm_y, hex_color):
        self._markers.append((norm_x, norm_y, hex_color))
        self._redraw()

    def addStrokePath(self, norm_points, hex_color, radius):
        self._strokes_paths.append((norm_points, hex_color, radius))
        self._redraw()

    def clearOverlays(self):
        self._markers.clear()
        self._strokes_paths.clear()
        self._current_stroke.clear()
        self._redraw()

    def _display_rect(self):
        if not self._base_pm:
            return QRect(0, 0, self.width(), self.height())
        iw, ih = self._base_pm.width(), self._base_pm.height()
        cw, ch = self.width(), self.height()
        scale  = min(cw / iw, ch / ih)
        pw, ph = int(iw * scale), int(ih * scale)
        return QRect((cw - pw) // 2, (ch - ph) // 2, pw, ph)

    def _redraw(self):
        if not self._base_pm:
            return
        rect   = self._display_rect()
        scaled = self._base_pm.scaled(
            rect.width(), rect.height(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        canvas_pm = QPixmap(self.width(), self.height())
        canvas_pm.fill(QColor(0x1a, 0x1a, 0x1a))
        p = QPainter(canvas_pm)
        p.drawPixmap(rect.x(), rect.y(), scaled)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Çizilmiş fırça yolları (silik alan + kesikli merkez çizgisi + başlangıç/bitiş düğümleri)
        if self.show_guides:
            for s_idx, (pts, col, rad) in enumerate(self._strokes_paths):
                if len(pts) > 1:
                    qc = QColor(col)
                    is_selected = (s_idx == self.selected_stroke_idx)

                    # 1. Silik fırça genişliği alanı (tül gibi saydam)
                    alpha_area = 70 if is_selected else 35
                    pen_w = max(2.0, (rad * 2.0) * (rect.width() / CANVAS_W))
                    pen_area = QPen(QColor(qc.red(), qc.green(), qc.blue(), alpha_area), pen_w,
                                    Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                    p.setPen(pen_area)
                    path = QPainterPath()
                    p_start = QPointF(rect.x() + pts[0][0] * rect.width(), rect.y() + pts[0][1] * rect.height())
                    path.moveTo(p_start)
                    for pt in pts[1:]:
                        path.lineTo(QPointF(rect.x() + pt[0] * rect.width(), rect.y() + pt[1] * rect.height()))
                    p.drawPath(path)

                    # 2. Kesikli merkez çizgisi (seçiliyse parlak altın, değilse zarif kontrastlı)
                    if is_selected:
                        pen_center = QPen(QColor(255, 215, 0, 240), 2.2, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap)
                    else:
                        pen_center = QPen(QColor(qc.red(), qc.green(), qc.blue(), 190), 1.5, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap)
                    p.setPen(pen_center)
                    p.drawPath(path)

                    # 3. Başlangıç (yeşil halka) ve Bitiş (kırmızı halka) düğümleri
                    p_end = QPointF(rect.x() + pts[-1][0] * rect.width(), rect.y() + pts[-1][1] * rect.height())
                    # Başlangıç düğümü (fırçanın değdiği yer)
                    p.setPen(QPen(QColor(0, 0, 0, 200), 1.5))
                    p.setBrush(QBrush(QColor(50, 205, 50, 220)))  # Yeşil
                    p.drawEllipse(p_start, 4.0, 4.0)

                    # Bitiş düğümü (fırçanın kalktığı yer)
                    p.setBrush(QBrush(QColor(255, 69, 0, 220)))   # Turuncu-Kırmızı
                    p.drawEllipse(p_end, 4.0, 4.0)

        # Şu an çizilmekte olan fırça darbesi (canlı el hareketi)
        if self._is_drawing and len(self._current_stroke) > 1:
            rc, gc, bc = self.cur_brush_color
            pen_w = max(2.0, (self.cur_brush_radius * 2.0) * (rect.width() / CANVAS_W))
            pen = QPen(QColor(rc, gc, bc, 120), pen_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            path = QPainterPath()
            p0 = self._current_stroke[0]
            path.moveTo(QPointF(rect.x() + (p0[0] / CANVAS_W) * rect.width(), rect.y() + (p0[1] / CANVAS_H) * rect.height()))
            for pt in self._current_stroke[1:]:
                path.lineTo(QPointF(rect.x() + (pt[0] / CANVAS_W) * rect.width(), rect.y() + (pt[1] / CANVAS_H) * rect.height()))
            p.drawPath(path)

        # Damla işaretçileri (küçük şık halkalar)
        if self.show_guides:
            for d_idx, (nx, ny, col) in enumerate(self._markers):
                wx = rect.x() + int(nx * rect.width())
                wy = rect.y() + int(ny * rect.height())
                fill = QColor(col)
                is_selected = (d_idx == self.selected_marker_idx)
                if is_selected:
                    p.setPen(QPen(QColor(255, 215, 0, 240), 2.5))
                    p.setBrush(QBrush(fill))
                    p.drawEllipse(wx - 7, wy - 7, 14, 14)
                else:
                    p.setPen(QPen(QColor(0, 0, 0, 180), 1.5))
                    p.setBrush(QBrush(fill))
                    p.drawEllipse(wx - 5, wy - 5, 10, 10)

        p.end()
        self.setPixmap(canvas_pm)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._redraw()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        rect = self._display_rect()
        pos  = event.position()
        lx   = pos.x() - rect.x()
        ly   = pos.y() - rect.y()
        if 0 <= lx <= rect.width() and 0 <= ly <= rect.height():
            cx = lx / rect.width() * CANVAS_W
            cy = ly / rect.height() * CANVAS_H

            if self.mode == "DROP":
                self.dropPlaced.emit(cx, cy)
            elif self.mode == "STROKE":
                self._is_drawing = True
                self._current_stroke = [(cx, cy)]
                self._redraw()

    def mouseMoveEvent(self, event):
        if self._is_drawing and self.mode == "STROKE":
            rect = self._display_rect()
            pos  = event.position()
            lx   = pos.x() - rect.x()
            ly   = pos.y() - rect.y()
            cx = max(0.0, min(float(CANVAS_W), lx / rect.width() * CANVAS_W))
            cy = max(0.0, min(float(CANVAS_H), ly / rect.height() * CANVAS_H))

            # Son noktayla arasında en az 4 piksel varsa ekle
            last_p = self._current_stroke[-1]
            if math.hypot(cx - last_p[0], cy - last_p[1]) >= 4.0:
                self._current_stroke.append((cx, cy))
                self._redraw()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._is_drawing and self.mode == "STROKE":
            self._is_drawing = False
            if len(self._current_stroke) >= 2:
                self.strokeCompleted.emit(list(self._current_stroke))
            self._current_stroke = []
            self._redraw()


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("💧 Sulu Boya Simulasyonu (Firca & Damla)")
        self.resize(1380, 840)

        self._drops     = []
        self._strokes   = []
        self._history   = []  # ("stroke", obj) veya ("drop", obj)
        self._worker    = None
        self._cur_color = (40, 120, 200)
        self._save_path = ""

        self._build_ui()
        self._update_color_btn()

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        left = QWidget()
        lv   = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        lv.setSpacing(6)

        # Mod Seçici Butonlar
        mode_box = QHBoxLayout()
        self.rb_stroke = QRadioButton("🖌️ Firca Modu (Surukle)")
        self.rb_drop   = QRadioButton("💧 Damla Modu (Tikla)")
        self.rb_stroke.setChecked(True)
        self.rb_stroke.toggled.connect(self._on_mode_changed)

        self.cb_guides = QCheckBox("📐 Kilavuzlari Goster")
        self.cb_guides.setChecked(True)
        self.cb_guides.toggled.connect(self._on_guides_toggled)

        mode_box.addWidget(self.rb_stroke)
        mode_box.addWidget(self.rb_drop)
        mode_box.addSpacing(20)
        mode_box.addWidget(self.cb_guides)
        mode_box.addStretch()
        lv.addLayout(mode_box)

        self.canvas = CanvasWidget()
        self.canvas.dropPlaced.connect(self._on_drop_placed)
        self.canvas.strokeCompleted.connect(self._on_stroke_completed)
        lv.addWidget(self.canvas)

        btn_row = QHBoxLayout()
        self.btn_preview = QPushButton("Onizleme")
        self.btn_save    = QPushButton("Kaydet (Tam)")
        self.btn_undo    = QPushButton("Geri Al")
        self.btn_reset   = QPushButton("Sifirla")

        self.btn_preview.clicked.connect(self._render_preview)
        self.btn_save.clicked.connect(self._render_full)
        self.btn_undo.clicked.connect(self._undo)
        self.btn_reset.clicked.connect(self._reset)

        for btn in (self.btn_preview, self.btn_save,
                    self.btn_undo, self.btn_reset):
            btn.setMinimumHeight(36)
            btn_row.addWidget(btn)

        lv.addLayout(btn_row)
        splitter.addWidget(left)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(330)
        scroll.setMaximumWidth(380)

        right = QWidget()
        rv    = QVBoxLayout(right)
        rv.setContentsMargins(8, 8, 8, 8)
        rv.setSpacing(8)

        # Renk Seçici (Üst ortak alan)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Aktif Renk:"))
        self.btn_color = QPushButton()
        self.btn_color.setFixedSize(90, 28)
        self.btn_color.clicked.connect(self._pick_color)
        crow.addWidget(self.btn_color)
        crow.addStretch()
        rv.addLayout(crow)

        tabs = QTabWidget()
        tabs.addTab(self._build_brush_tab(), "🖌️ Firca")
        tabs.addTab(self._build_drop_tab(),  "💧 Damla")
        tabs.addTab(self._build_env_tab(),   "🌊 Ortam")
        rv.addWidget(tabs)

        rv.addWidget(self._build_list_panel())
        rv.addStretch()

        scroll.setWidget(right)
        splitter.addWidget(scroll)
        splitter.setSizes([980, 360])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Hazir - Firca ile cizin, ardindan Onizleme'ye basin.")

    def _build_brush_tab(self):
        w  = QWidget()
        vl = QVBoxLayout(w)
        vl.setSpacing(5)
        vl.setContentsMargins(4, 6, 4, 4)

        self.p_br_radius   = ParamRow("Firca Capi:",     5, 150, 30, decimals=0)
        self.p_br_intens   = ParamRow("Yogunluk:",     0.1, 1.0, 0.85)
        self.p_br_edge     = ParamRow("Kenar Halkasi:",0.0, 1.0, 0.60)
        self.p_br_dry      = ParamRow("Kuru Firca:",   0.0, 1.0, 0.45)
        self.p_br_bristle  = ParamRow("Kil Lifleri:",  0.0, 1.0, 0.35)
        self.p_br_paper    = ParamRow("Kagit Etkisi:", 0.0, 0.5, 0.12)

        self.p_br_radius.valueChanged.connect(self._sync_brush_radius)

        for pw in (self.p_br_radius, self.p_br_intens, self.p_br_edge,
                   self.p_br_dry, self.p_br_bristle, self.p_br_paper):
            vl.addWidget(pw)

        vl.addStretch()
        return w

    def _build_drop_tab(self):
        w  = QWidget()
        vl = QVBoxLayout(w)
        vl.setSpacing(4)
        vl.setContentsMargins(4, 6, 4, 4)

        self.p_radius   = ParamRow("Yaricap:",      10, 400, 150, decimals=0)
        self.p_intens   = ParamRow("Yogunluk:",     0.1, 1.0, 0.80)
        self.p_edge_dk  = ParamRow("Kenar:",        0.0, 1.0, 0.60)
        self.p_feather  = ParamRow("Feathering:",   0.0, 1.0, 0.40)
        self.p_paper_ix = ParamRow("Kagit Etki:",   0.0, 0.5, 0.12)

        for pw in (self.p_radius, self.p_intens, self.p_edge_dk,
                   self.p_feather, self.p_paper_ix):
            vl.addWidget(pw)

        self.splash_grp = QGroupBox("Sicrama")
        self.splash_grp.setCheckable(True)
        self.splash_grp.setChecked(True)
        sg = QVBoxLayout(self.splash_grp)
        self.p_spl_count      = ParamRow("Sayi:",         1,  40,  8,    decimals=0)
        self.p_spl_scat_min   = ParamRow("Dagilim min:",  0.2, 2.5, 1.15)
        self.p_spl_scat_max   = ParamRow("Dagilim max:",  0.5, 4.0, 1.80)
        self.p_spl_intens     = ParamRow("Yogunluk:",     0.1, 1.0, 0.65)
        self.p_spl_wet_react  = ParamRow("Zemin Etkisi:", 0.0, 1.0, 0.85)
        self.p_spl_dispersion = ParamRow("Ciceklenme:",   1.0, 2.5, 1.70)
        for pw in (self.p_spl_count, self.p_spl_scat_min,
                   self.p_spl_scat_max, self.p_spl_intens,
                   self.p_spl_wet_react, self.p_spl_dispersion):
            sg.addWidget(pw)
        vl.addWidget(self.splash_grp)

        self.bloom_grp = QGroupBox("Islak Bloom")
        self.bloom_grp.setCheckable(True)
        self.bloom_grp.setChecked(True)
        bg = QVBoxLayout(self.bloom_grp)
        self.p_bloom_str  = ParamRow("Guc:",   0.0, 1.0, 0.50)
        self.p_bloom_size = ParamRow("Boyut:", 0.05, 0.5, 0.20)
        for pw in (self.p_bloom_str, self.p_bloom_size):
            bg.addWidget(pw)
        vl.addWidget(self.bloom_grp)

        vl.addStretch()
        return w

    def _build_env_tab(self):
        w  = QWidget()
        vl = QVBoxLayout(w)
        vl.setSpacing(6)
        vl.setContentsMargins(4, 6, 4, 4)

        pg  = QGroupBox("Kagit Dokusu")
        pgl = QVBoxLayout(pg)
        self.p_pap_scale   = ParamRow("Olcek:",    50, 400, 200, decimals=0)
        self.p_pap_octaves = ParamRow("Oktav:",     1,   8,   6, decimals=0)
        self.p_pap_intens  = ParamRow("Yogunluk:", 0.0, 0.4, 0.12)
        for pw in (self.p_pap_scale, self.p_pap_octaves, self.p_pap_intens):
            pgl.addWidget(pw)
        vl.addWidget(pg)

        flg = QGroupBox("Akis Simulasyonu")
        fll = QVBoxLayout(flg)
        self.p_fl_iter  = ParamRow("Iterasyon:",    1, 30,  10, decimals=0)
        self.p_fl_diff  = ParamRow("Difuzyon:",  0.01, 0.5, 0.12)
        self.p_fl_turb  = ParamRow("Turbulans:", 0.0,  5.0, 1.5)
        self.p_fl_grav  = ParamRow("Yer Cekimi:", 0.0, 1.0, 0.3)
        self.p_fl_angle = ParamRow("Aci (deg):",   0, 360,  90, decimals=0)
        for pw in (self.p_fl_iter, self.p_fl_diff, self.p_fl_turb,
                   self.p_fl_grav, self.p_fl_angle):
            fll.addWidget(pw)
        vl.addWidget(flg)

        dg  = QGroupBox("Kuruma")
        dgl = QVBoxLayout(dg)
        self.p_dry_val = ParamRow("Parlaklik:", -0.3, 0.3,  0.05)
        self.p_dry_sat = ParamRow("Doygunluk:", -0.5, 0.5, -0.08)
        for pw in (self.p_dry_val, self.p_dry_sat):
            dgl.addWidget(pw)
        vl.addWidget(dg)

        vl.addStretch()
        return w

    def _build_list_panel(self):
        grp = QGroupBox("Cizim Gecmisi")
        lay = QVBoxLayout(grp)
        self.item_list = QListWidget()
        self.item_list.setMaximumHeight(160)
        self.item_list.setMinimumHeight(60)
        self.item_list.currentRowChanged.connect(self._on_item_selected)
        lay.addWidget(self.item_list)
        del_btn = QPushButton("Seciliyi Sil")
        del_btn.clicked.connect(self._delete_selected)
        lay.addWidget(del_btn)
        return grp

    def _on_item_selected(self, row):
        if row < 0 or row >= len(self._history):
            self.canvas.selected_stroke_idx = -1
            self.canvas.selected_marker_idx = -1
            self.canvas._redraw()
            return
        kind, obj = self._history[row]
        if kind == "stroke":
            self.canvas.selected_stroke_idx = self._strokes.index(obj) if obj in self._strokes else -1
            self.canvas.selected_marker_idx = -1
        elif kind == "drop":
            self.canvas.selected_marker_idx = self._drops.index(obj) if obj in self._drops else -1
            self.canvas.selected_stroke_idx = -1
        self.canvas._redraw()

    def _on_mode_changed(self):
        if self.rb_stroke.isChecked():
            self.canvas.mode = "STROKE"
            self.status_bar.showMessage("Firca Modu: Tuval uzerinde fareyi surukleyerek cizin.")
        else:
            self.canvas.mode = "DROP"
            self.status_bar.showMessage("Damla Modu: Tuvale tiklayarak damla birakin.")

    def _sync_brush_radius(self, val):
        self.canvas.cur_brush_radius = float(val)

    def _pick_color(self):
        qc = QColorDialog.getColor(QColor(*self._cur_color), self, "Renk Sec")
        if qc.isValid():
            self._cur_color = (qc.red(), qc.green(), qc.blue())
            self.canvas.cur_brush_color = self._cur_color
            self._update_color_btn()

    def _update_color_btn(self):
        r, g, b = self._cur_color
        self.btn_color.setStyleSheet(f"background-color:rgb({r},{g},{b}); border:1px solid #333;")
        self.btn_color.setText(f"#{r:02X}{g:02X}{b:02X}")
        self.canvas.cur_brush_color = self._cur_color

    def _on_stroke_completed(self, points):
        r, g, b = self._cur_color
        rad = self.p_br_radius.value()
        stroke = {
            "points": points,
            "color": (r, g, b),
            "radius": rad,
            "intensity": self.p_br_intens.value(),
            "edge_darkening": self.p_br_edge.value(),
            "dry_brush": self.p_br_dry.value(),
            "bristle_strength": self.p_br_bristle.value(),
            "paper_interaction": self.p_br_paper.value(),
        }
        self._strokes.append(stroke)
        self._history.append(("stroke", stroke))

        # Önizleme overlayine ekle
        norm_pts = [(p[0] / CANVAS_W, p[1] / CANVAS_H) for p in points]
        self.canvas.addStrokePath(norm_pts, f"#{r:02X}{g:02X}{b:02X}", rad)

        item = QListWidgetItem(f"🖌️ Firca ({len(points)} nokta)  R:{int(rad)}  #{r:02X}{g:02X}{b:02X}")
        item.setForeground(QColor(r, g, b))
        self.item_list.addItem(item)
        self.item_list.scrollToBottom()
        self.status_bar.showMessage(f"Firca darbesi eklendi - 'Onizleme'ye basarak sulu boya akisini gorun.")

    def _on_drop_placed(self, cx, cy):
        r, g, b = self._cur_color
        drop = {
            "x": cx, "y": cy,
            "color": (r, g, b),
            "radius":              int(self.p_radius.value()),
            "intensity":           self.p_intens.value(),
            "edge_darkening":      self.p_edge_dk.value(),
            "feathering":          self.p_feather.value(),
            "paper_interaction":   self.p_paper_ix.value(),
            "splatter_enabled":    self.splash_grp.isChecked(),
            "splatter_count":      int(self.p_spl_count.value()),
            "scatter_min":         self.p_spl_scat_min.value(),
            "scatter_max":         self.p_spl_scat_max.value(),
            "splatter_intensity":  self.p_spl_intens.value(),
            "splatter_wet_react":  self.p_spl_wet_react.value(),
            "splatter_dispersion": self.p_spl_dispersion.value(),
            "bloom_enabled":       self.bloom_grp.isChecked(),
            "bloom_strength":      self.p_bloom_str.value(),
            "bloom_size":          self.p_bloom_size.value(),
        }
        self._drops.append(drop)
        self._history.append(("drop", drop))

        self.canvas.addMarker(cx / CANVAS_W, cy / CANVAS_H, f"#{r:02X}{g:02X}{b:02X}")

        item = QListWidgetItem(f"💧 Damla X:{int(cx):3d} Y:{int(cy):3d} R:{drop['radius']:3d} #{r:02X}{g:02X}{b:02X}")
        item.setForeground(QColor(r, g, b))
        self.item_list.addItem(item)
        self.item_list.scrollToBottom()
        self.status_bar.showMessage(f"Damla eklendi - 'Onizleme'ye basin.")

    def _undo(self):
        if not self._history:
            return
        kind, obj = self._history.pop()
        if kind == "drop":
            self._drops.remove(obj)
        elif kind == "stroke":
            self._strokes.remove(obj)
        self.item_list.takeItem(self.item_list.count() - 1)

        # Overlay'leri yeniden olustur
        self.canvas.clearOverlays()
        for d in self._drops:
            rc, gc, bc = d["color"]
            self.canvas.addMarker(d["x"] / CANVAS_W, d["y"] / CANVAS_H, f"#{rc:02X}{gc:02X}{bc:02X}")
        for st in self._strokes:
            rc, gc, bc = st["color"]
            npts = [(p[0] / CANVAS_W, p[1] / CANVAS_H) for p in st["points"]]
            self.canvas.addStrokePath(npts, f"#{rc:02X}{gc:02X}{bc:02X}", st["radius"])

        self.status_bar.showMessage(f"Geri alindi.")

    def _reset(self):
        if QMessageBox.question(
            self, "Sifirla", "Tum cizimler silinecek. Emin misin?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        self._drops.clear()
        self._strokes.clear()
        self._history.clear()
        self.canvas.clearOverlays()
        self.canvas._init_placeholder()
        self.item_list.clear()
        self.status_bar.showMessage("Tuval sifirlandi.")

    def _delete_selected(self):
        row = self.item_list.currentRow()
        if row < 0 or row >= len(self._history):
            return
        kind, obj = self._history.pop(row)
        if kind == "drop":
            self._drops.remove(obj)
        elif kind == "stroke":
            self._strokes.remove(obj)
        self.item_list.takeItem(row)

        self.canvas.clearOverlays()
        for d in self._drops:
            rc, gc, bc = d["color"]
            self.canvas.addMarker(d["x"] / CANVAS_W, d["y"] / CANVAS_H, f"#{rc:02X}{gc:02X}{bc:02X}")
        for st in self._strokes:
            rc, gc, bc = st["color"]
            npts = [(p[0] / CANVAS_W, p[1] / CANVAS_H) for p in st["points"]]
            self.canvas.addStrokePath(npts, f"#{rc:02X}{gc:02X}{bc:02X}", st["radius"])

        self.status_bar.showMessage(f"Oge silindi.")

    def _build_config(self, preview):
        return {
            "preview": preview,
            "strokes": self._strokes,
            "drops":   self._drops,
            "paper": {
                "scale":     self.p_pap_scale.value(),
                "octaves":   max(1, int(self.p_pap_octaves.value())),
                "intensity": self.p_pap_intens.value(),
            },
            "flow": {
                "iterations":       max(1, int(self.p_fl_iter.value())),
                "diffusion_rate":   self.p_fl_diff.value(),
                "turbulence":       self.p_fl_turb.value(),
                "gravity_strength": self.p_fl_grav.value(),
                "gravity_angle":    self.p_fl_angle.value(),
            },
            "drying": {
                "value_shift":      self.p_dry_val.value(),
                "saturation_shift": self.p_dry_sat.value(),
            },
        }

    def _start_render(self, preview):
        if self._worker and self._worker.isRunning():
            self.status_bar.showMessage("Render hala calisiyor...")
            return
        self._set_buttons_enabled(False)
        self._worker = RenderWorker(self._build_config(preview))
        self._worker.statusMsg.connect(self.status_bar.showMessage)
        self._worker.finished.connect(self._on_render_done)
        self._worker.start()

    def _render_preview(self):
        self._start_render(preview=True)

    def _render_full(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "PNG Olarak Kaydet", "output_paper.png", "PNG (*.png)"
        )
        if not path:
            return
        self._save_path = path
        self._start_render(preview=False)

    def _on_guides_toggled(self, checked):
        self.canvas.show_guides = checked
        self.canvas._redraw()

    def _on_render_done(self, result):
        self._set_buttons_enabled(True)
        if result is None:
            return
        arr, was_preview = result
        self.canvas.setImage(arr)

        # Render sonucu tuval altligina yerlestirilir, kilavuzlar ise silinmeyip
        # kullanicinin cizim rotasini gostererek ekranda kalmaya devam eder!
        self.canvas._redraw()

        if not was_preview and self._save_path:
            out_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self._save_path, out_bgr)
            self.status_bar.showMessage(f"Kaydedildi -> {self._save_path}")

    def _set_buttons_enabled(self, enabled):
        for btn in (self.btn_preview, self.btn_save,
                    self.btn_undo, self.btn_reset):
            btn.setEnabled(enabled)


def _apply_dark_theme(app):
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor(40,  40,  40))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Base,            QColor(30,  30,  30))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(50,  50,  50))
    p.setColor(QPalette.ColorRole.Text,            QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Button,          QColor(58,  58,  58))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Highlight,       QColor(42,  130, 218))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(0,   0,   0))
    app.setPalette(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    _apply_dark_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())