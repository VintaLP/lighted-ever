import os
import subprocess
import argparse
import re

def fix_permissions(path):
    """Setzt rekursiv die Rechte, damit jeder den Ordner lesen/schreiben kann."""
    try:
        # Ordner für alle lesbar und schreibbar machen (777)
        subprocess.run(["chmod", "-R", "777", path], check=False, stderr=subprocess.DEVNULL)
        # Besitzer auf Standard-User ändern (schlägt ohne root manchmal fehl, stört aber nicht)
        subprocess.run(["chown", "-R", "1000:1000", path], check=False, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Run train.py on multiple folders sequentially.")
    parser.add_argument("--input_base", type=str, required=True, help="Base directory containing dataset folders")
    parser.add_argument("--output_base", type=str, required=True, help="Base directory for output models")
    parser.add_argument("--summary_file", type=str, required=True, help="Base path to the output text file")
    args = parser.parse_args()

    # Zielordner sicherstellen
    output_dir = os.path.dirname(os.path.abspath(args.summary_file))
    os.makedirs(output_dir, exist_ok=True)
    
    # Dateinamen für die 4 Kategorien generieren
    base_name, ext = os.path.splitext(os.path.basename(args.summary_file))
    files = {
        "lighted": os.path.join(output_dir, f"{base_name}_lighted{ext}"),
        "unlit": os.path.join(output_dir, f"{base_name}_unlit{ext}"),
        "normals": os.path.join(output_dir, f"{base_name}_normals{ext}"),
        "brightness": os.path.join(output_dir, f"{base_name}_brightness{ext}")
    }

    if not os.path.exists(args.input_base):
        print(f"Fehler: Datensatz-Ordner {args.input_base} existiert nicht.")
        return

    dataset_folders = [f for f in os.listdir(args.input_base) if os.path.isdir(os.path.join(args.input_base, f))]
    dataset_folders.sort()

    # Header in alle Textdateien schreiben (falls sie noch nicht existieren)
    for path in files.values():
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write("Dataset_Name\tTest_L1\tTest_PSNR\tTest_Masked_L1\tTest_Masked_PSNR\n")

    print(f"Gefundene Datensätze: {len(dataset_folders)} -> {dataset_folders}")

    for folder in dataset_folders:
        print(f"\n{'='*50}\nStarte Training für: {folder}\n{'='*50}")
        
        dataset_path = os.path.join(args.input_base, folder)
        output_path = os.path.join(args.output_base, folder)

        cmd = ["python", "train.py", "-s", dataset_path, "-m", output_path, "--eval"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Dictionary zum Zwischenspeichern der aktuellen Werte
        metrics = {
            "lighted": {"l1": "N/A", "psnr": "N/A"}, "lighted_masked": {"l1": "N/A", "psnr": "N/A"},
            "unlit": {"l1": "N/A", "psnr": "N/A"}, "unlit_masked": {"l1": "N/A", "psnr": "N/A"},
            "normals": {"l1": "N/A", "psnr": "N/A"}, "normals_masked": {"l1": "N/A", "psnr": "N/A"},
            "brightness": {"l1": "N/A", "psnr": "N/A"}, "brightness_masked": {"l1": "N/A", "psnr": "N/A"}
        }

        # Regex um den Namen nach 'test' und die Werte auszulesen
        regex_eval = re.compile(r"Evaluating test(.*): L1 ([\d\.]+) PSNR ([\d\.]+)")

        for line in process.stdout:
            print(line, end="") 
            
            match = regex_eval.search(line)
            if match:
                suffix = match.group(1) # z.B. "_unlit_masked" oder "" (leer) für normales lighted
                l1_val = match.group(2)
                psnr_val = match.group(3)
                
                # Werte dem richtigen Typ zuordnen
                if suffix == "":
                    metrics["lighted"]["l1"], metrics["lighted"]["psnr"] = l1_val, psnr_val
                elif suffix == "_masked":
                    metrics["lighted_masked"]["l1"], metrics["lighted_masked"]["psnr"] = l1_val, psnr_val
                elif suffix == "_unlit":
                    metrics["unlit"]["l1"], metrics["unlit"]["psnr"] = l1_val, psnr_val
                elif suffix == "_unlit_masked":
                    metrics["unlit_masked"]["l1"], metrics["unlit_masked"]["psnr"] = l1_val, psnr_val
                elif suffix == "_normals":
                    metrics["normals"]["l1"], metrics["normals"]["psnr"] = l1_val, psnr_val
                elif suffix == "_normals_masked":
                    metrics["normals_masked"]["l1"], metrics["normals_masked"]["psnr"] = l1_val, psnr_val
                elif suffix == "_brightness":
                    metrics["brightness"]["l1"], metrics["brightness"]["psnr"] = l1_val, psnr_val
                elif suffix == "_brightness_masked":
                    metrics["brightness_masked"]["l1"], metrics["brightness_masked"]["psnr"] = l1_val, psnr_val

        process.wait()

        # Ergebnisse am Ende des Trainings in die jeweiligen Dateien anhängen
        with open(files["lighted"], "a") as f:
            f.write(f"{folder}\t{metrics['lighted']['l1']}\t{metrics['lighted']['psnr']}\t{metrics['lighted_masked']['l1']}\t{metrics['lighted_masked']['psnr']}\n")
            
        with open(files["unlit"], "a") as f:
            f.write(f"{folder}\t{metrics['unlit']['l1']}\t{metrics['unlit']['psnr']}\t{metrics['unlit_masked']['l1']}\t{metrics['unlit_masked']['psnr']}\n")

        with open(files["normals"], "a") as f:
            f.write(f"{folder}\t{metrics['normals']['l1']}\t{metrics['normals']['psnr']}\t{metrics['normals_masked']['l1']}\t{metrics['normals_masked']['psnr']}\n")
            
        with open(files["brightness"], "a") as f:
            f.write(f"{folder}\t{metrics['brightness']['l1']}\t{metrics['brightness']['psnr']}\t{metrics['brightness_masked']['l1']}\t{metrics['brightness_masked']['psnr']}\n")

        fix_permissions(output_path)

        print(f"\n--- Training für {folder} abgeschlossen. Alle 4 Textdateien wurden aktualisiert ---\n")

if __name__ == "__main__":
    main()