from __future__ import annotations


def build_enrollment_panel(ft, on_pick_samples, on_enroll, selected_samples_control):
    speaker_id = ft.TextField(label="Speaker ID", dense=True)
    display_name = ft.TextField(label="Display name", dense=True)

    def submit(_):
        on_enroll(speaker_id.value.strip(), display_name.value.strip())

    return ft.Column(
        controls=[
            ft.Text("Voice Enrollment", size=18, weight=ft.FontWeight.W_600),
            speaker_id,
            display_name,
            ft.Row(
                controls=[
                    ft.ElevatedButton("Pick samples", icon=ft.Icons.UPLOAD_FILE, on_click=on_pick_samples),
                    ft.ElevatedButton("Enroll", icon=ft.Icons.PERSON_ADD, on_click=submit),
                ],
                wrap=True,
            ),
            selected_samples_control,
        ],
        spacing=10,
    )
