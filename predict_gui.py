import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import io
from rdkit import Chem
from rdkit.Chem import Draw

# あなたの推論ユーティリティ（GenericGNNPredictor + load_predictor_generic）
from gnn_infer import load_predictor_generic


class PredictApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GNN Predictor (ZINC + NREL)")
        self.resize(900, 650)

        # ---------- Load models once ----------
        try:
            self.pred_zinc = load_predictor_generic(
                dataset_name="zinc",
                save_root="models",
                ckpt_name="checkpoint.pt",
                ad_dir='models/ad_artifacts/zinc',
                )
        except Exception as e:
            self.pred_zinc = None
            QMessageBox.critical(self, "Load Error", f"Failed to load ZINC predictor:\n{e}")

        try:
            self.pred_nrel = load_predictor_generic(
                dataset_name="nrel",
                save_root="models",
                ckpt_name="checkpoint.pt",
                ad_dir="models/ad_artifacts/nrel",
                )
        except Exception as e:
            self.pred_nrel = None
            QMessageBox.critical(self, "Load Error", f"Failed to load NREL predictor:\n{e}")

        # ---------- UI ----------
        main = QVBoxLayout()

        # input row
        row = QHBoxLayout()
        row.addWidget(QLabel("SMILES:"))
        self.smiles_edit = QLineEdit()
        self.smiles_edit.setPlaceholderText("e.g., CCO")
        row.addWidget(self.smiles_edit, stretch=1)

        self.btn_predict = QPushButton("Predict (ZINC + NREL)")
        self.btn_predict.clicked.connect(self.on_predict)
        row.addWidget(self.btn_predict)

        main.addLayout(row)

        # --- molecule viewer (under SMILES input) ---
        self.mol_label = QLabel("Molecule preview will appear here.")
        self.mol_label.setAlignment(Qt.AlignCenter)
        self.mol_label.setFixedHeight(180)          # 好みで調整（例: 160〜220）
        self.mol_label.setStyleSheet("border: 1px solid #cccccc; background: #ffffff;")
        main.addWidget(self.mol_label)

        # outputs
        out_row = QHBoxLayout()

        self.box_zinc = QGroupBox("ZINC (logP / qed / SAS)")
        zinc_layout = QVBoxLayout()
        self.out_zinc = QPlainTextEdit()
        self.out_zinc.setReadOnly(True)
        zinc_layout.addWidget(self.out_zinc)
        self.box_zinc.setLayout(zinc_layout)
        self.pb_zinc = QProgressBar()
        self.pb_zinc.setRange(0, 100)
        zinc_layout.addWidget(self.pb_zinc)

        self.box_nrel = QGroupBox("NREL (8 targets)")
        nrel_layout = QVBoxLayout()
        self.out_nrel = QPlainTextEdit()
        self.out_nrel.setReadOnly(True)
        nrel_layout.addWidget(self.out_nrel)
        self.box_nrel.setLayout(nrel_layout)
        self.pb_nrel = QProgressBar()
        self.pb_nrel.setRange(0, 100)
        nrel_layout.addWidget(self.pb_nrel)

        out_row.addWidget(self.box_zinc, stretch=1)
        out_row.addWidget(self.box_nrel, stretch=1)

        main.addLayout(out_row)

        # status
        self.status = QLabel("Ready.")
        main.addWidget(self.status)

        self.setLayout(main)

        # Enter key triggers predict
        self.smiles_edit.returnPressed.connect(self.on_predict)

        # keep original pixmap for resize scaling
        self._mol_pixmap_raw = None

    def _smiles_to_pixmap(self, smiles: str, size=(520, 170)) -> QPixmap:
        """
        RDKitでSMILESを描画し、QtのQPixmapに変換して返す。
        size: (width, height)
        """
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')  # suppress RDKit warnings

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return QPixmap()

        # RDKit -> PIL image
        img = Draw.MolToImage(mol, size=size)

        # PIL -> PNG bytes -> QImage -> QPixmap
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        qimg = QImage.fromData(buf.getvalue(), "PNG")
        return QPixmap.fromImage(qimg)

    def _format_result(self, res: dict) -> str:
        """
        res is the dict returned by predictor.predict_smiles(...)
        """
        if not res.get("ok", False):
            err = res.get("error", "unknown error")
            smi_in = res.get("smiles_in", "")
            smi_used = res.get("smiles_used", "")
            return f"ERROR\nsmiles_in : {smi_in}\nsmiles_used: {smi_used}\nreason    : {err}\n"

        smi_in = res.get("smiles_in", "")
        smi_used = res.get("smiles_used", "")
        pred_raw = res.get("pred_raw", {})

        lines = []
        lines.append(f"smiles_in  : {smi_in}")
        lines.append(f"smiles_used: {smi_used}")
        lines.append("")

        # AD info (if present)
        ad = res.get("ad", None)
        if isinstance(ad, dict) and ad.get("ok", False):
            ds = res.get("dataset", "MODEL")
            label = ad.get("label", "")
            score = float(ad.get("score", float("nan")))
            thr   = float(ad.get("threshold", float("nan")))

            if label == "out":
                lines.append("⚠⚠⚠ OUT OF AD ⚠⚠⚠")      # ←強調行
            lines.append(f"AD ({ds.upper()}): {label}  [score={score:.3f}, thr={thr:.3f}]")
            lines.append("")

        for k, v in pred_raw.items():
            # show as 6 significant digits
            try:
                lines.append(f"{k:>10s} : {float(v):.6g}")
            except Exception:
                lines.append(f"{k:>10s} : {v}")
        return "\n".join(lines) + "\n"
    
    def _set_ad_background(self, text_edit: QPlainTextEdit, res: dict):
        ad = res.get("ad", None)
        if not isinstance(ad, dict) or not ad.get("ok", False):
            text_edit.setStyleSheet("")
            return

        label = ad.get("label", "")
        score = float(ad.get("score", 0.0))
        thr   = float(ad.get("threshold", 0.0))

        # gray zone: [0.9*thr, thr]
        gray_low = 0.9 * thr

        if label == "out":
            text_edit.setStyleSheet("QPlainTextEdit { background-color: #ffe8e8; }")  # red
        elif score >= gray_low:
            text_edit.setStyleSheet("QPlainTextEdit { background-color: #fff7cc; }")  # yellow
        else:
            text_edit.setStyleSheet("QPlainTextEdit { background-color: #e8f2ff; }")  # blue

    def on_predict(self):
        smi = self.smiles_edit.text().strip()
        if not smi:
            QMessageBox.warning(self, "Input Required", "Please enter a SMILES string.")
            return

        self.status.setText("Predicting...")
        self.btn_predict.setEnabled(False)

        try:
            res_z = {"ok": False}
            res_n = {"ok": False}

            # ZINC
            if self.pred_zinc is not None:
                res_z = self.pred_zinc.predict_smiles_with_ad(smi)
                self.out_zinc.setPlainText(self._format_result(res_z))
                self._set_ad_background(self.out_zinc, res_z)
                self._set_ad_progress(self.pb_zinc, res_z)
            else:
                self.out_zinc.setPlainText("ZINC predictor not loaded.\n")

            # NREL
            if self.pred_nrel is not None:
                res_n = self.pred_nrel.predict_smiles_with_ad(smi)
                self.out_nrel.setPlainText(self._format_result(res_n))
                self._set_ad_background(self.out_nrel, res_n)
                self._set_ad_progress(self.pb_nrel, res_n)
            else:
                self.out_nrel.setPlainText("NREL predictor not loaded.\n")

            # --- update molecule image ---
            smiles_used = None
            if self.pred_zinc is not None and res_z.get("ok", False):
                smiles_used = res_z.get("smiles_used", smi)
            elif self.pred_nrel is not None and res_n.get("ok", False):
                smiles_used = res_n.get("smiles_used", smi)
            else:
                smiles_used = smi

            gen_w = max(300, self.mol_label.width() * 2)
            gen_h = max(160, self.mol_label.height() * 2)
            pix = self._smiles_to_pixmap(smiles_used, size=(gen_w, gen_h))

            if pix.isNull():
                self._mol_pixmap_raw = None
                self.mol_label.setText("Invalid SMILES (cannot draw).")
                self.mol_label.setPixmap(QPixmap())
            else:
                self.mol_label.setText("")
                self._mol_pixmap_raw = pix

                # 現在のラベルサイズに合わせて表示
                w = max(10, self.mol_label.width() - 10)
                h = max(10, self.mol_label.height() - 10)
                self.mol_label.setPixmap(
                    pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )

            z_lab = self._ad_short(res_z)
            n_lab = self._ad_short(res_n)
            self.status.setText(f"Done.  ZINC: {z_lab} / NREL: {n_lab}")
            
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", str(e))
            self.status.setText("Error.")
        finally:
            self.btn_predict.setEnabled(True)

    def _set_ad_progress(self, pb: QProgressBar, res: dict):
        """
        AD score を progress bar で表示（thr=100%）
        """
        ad = res.get("ad", None)
        if not isinstance(ad, dict) or not ad.get("ok", False):
            pb.setValue(0)
            pb.setFormat("AD: N/A")
            pb.setStyleSheet("")
            return

        score = float(ad.get("score", 0.0))
        thr   = float(ad.get("threshold", 1.0))
        ratio = 0.0 if thr <= 0 else (score / thr)

        pb.setRange(0, 500)
        pct = int(min(500, max(0, round(100 * ratio))))
        pb.setValue(pct)
        pb.setFormat(f"AD score: {pct}%")

        # color by zone
        if ratio > 1.0:
            pb.setStyleSheet(
                "QProgressBar::chunk { background-color: #ff6b6b; }"
            )  # red
        elif ratio >= 0.9:
            pb.setStyleSheet(
                "QProgressBar::chunk { background-color: #ffd43b; }"
            )  # yellow
        else:
            pb.setStyleSheet(
                "QProgressBar::chunk { background-color: #4dabf7; }"
            )  # blue

    def _ad_short(self, res: dict) -> str:
        ad = res.get("ad", None)
        if isinstance(ad, dict) and ad.get("ok", False):
            return str(ad.get("label", "NA"))
        return "NA"

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if self._mol_pixmap_raw is None:
            return

        # label size (leave a small margin)
        w = max(10, self.mol_label.width() - 10)
        h = max(10, self.mol_label.height() - 10)

        pix = self._mol_pixmap_raw.scaled(
            w, h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.mol_label.setPixmap(pix)

def main():
    app = QApplication(sys.argv)
    w = PredictApp()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
