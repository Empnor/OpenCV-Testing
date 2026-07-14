import tkinter as tk
from tkinter import filedialog, messagebox
 
import cv2
import os
import shutil
 
import database
import face_utils
 
KNOWN_FACES_DIR = "known_faces"
 
 
class AdminApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition — Admin Dashboard")
        self.geometry("650x600")
 
        database.init_db()
        self.selected_files = []
 
        # --- Add person form ---
        form_frame = tk.Frame(self)
        form_frame.pack(fill="x", pady=10, padx=10)
 
        tk.Label(form_frame, text="Name:").grid(row=0, column=0, sticky="w")
        self.name_entry = tk.Entry(form_frame, width=30)
        self.name_entry.grid(row=0, column=1, padx=5)
 
        tk.Button(form_frame, text="Select Photos...", command=self.select_photos).grid(row=0, column=2, padx=5)
        self.files_label = tk.Label(form_frame, text="No photos selected")
        self.files_label.grid(row=1, column=0, columnspan=2, sticky="w")
 
        tk.Button(form_frame, text="Add Person", command=self.add_person).grid(row=2, column=0, pady=10, sticky="w")
 
        # --- People list ---
        tk.Label(self, text="Enrolled People", font=("Arial", 12, "bold")).pack(anchor="w", padx=10)
 
        list_frame = tk.Frame(self)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)
 
        self.people_listbox = tk.Listbox(list_frame)
        self.people_listbox.pack(side="left", fill="both", expand=True)
        scrollbar = tk.Scrollbar(list_frame, command=self.people_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.people_listbox.config(yscrollcommand=scrollbar.set)
 
        button_row = tk.Frame(self)
        button_row.pack(pady=5)
        tk.Button(button_row, text="Delete Selected Person", command=self.delete_selected).pack(side="left", padx=5)
        tk.Button(button_row, text="Refresh List", command=self.refresh_list).pack(side="left", padx=5)
 
        self.people_map = {}
        self.refresh_list()
 
    def select_photos(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if files:
            self.selected_files = list(files)
            self.files_label.config(text=f"{len(self.selected_files)} photo(s) selected")
 
    def add_person(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name.")
            return
        if not self.selected_files:
            messagebox.showerror("Error", "Please select at least one photo.")
            return
 
        person_id = database.add_person(name)
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
 
        added, skipped = 0, 0
        for src_path in self.selected_files:
            dst_path = os.path.join(person_dir, os.path.basename(src_path))
            shutil.copy(src_path, dst_path)
 
            image = cv2.imread(dst_path)
            embedding = face_utils.get_embedding(image) if image is not None else None
 
            if embedding is None:
                skipped += 1
                continue
 
            database.add_photo(person_id, dst_path, embedding)
            added += 1
 
        messagebox.showinfo("Done", f"Added {added} photo(s) for {name}." +
                             (f" Skipped {skipped} (no face detected)." if skipped else "") +
                             "\n\nIf the camera app is already running, click "
                             "'Reload Known Faces' there to pick up this change.")
 
        self.name_entry.delete(0, tk.END)
        self.selected_files = []
        self.files_label.config(text="No photos selected")
        self.refresh_list()
 
    def refresh_list(self):
        self.people_listbox.delete(0, tk.END)
        self.people_map = {}
        for person_id, name, photo_count in database.list_people():
            display = f"{name}  ({photo_count} photo(s))"
            self.people_listbox.insert(tk.END, display)
            self.people_map[display] = person_id
 
    def delete_selected(self):
        selection = self.people_listbox.curselection()
        if not selection:
            return
        display = self.people_listbox.get(selection[0])
        person_id = self.people_map[display]
 
        if messagebox.askyesno("Confirm", f"Delete {display}?"):
            database.delete_person(person_id)
            self.refresh_list()
 
 
if __name__ == "__main__":
    app = AdminApp()
    app.mainloop()