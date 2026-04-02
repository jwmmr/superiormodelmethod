It's really simple.

Run install packages, all it does is install numpy and pillow

meshgenerator_adaptive.py is the main script, run this.

It will ask for a path, you can drop an image into console, or paste a path to one.

It will then ask for quality, simply press enter for default, higher quality may use 2-4 so meshes for higher quality result. Default tries to be 1 mesh only.

You now have to upload the meshes into studio

Then go into the browser on Roblox creator dashboard, copy each mesh id, in order (left to right, reading direction), paste them individually into the console and press enter for each one.

Repeat the same for images (they are automatically uploaded with meshes)

Now open the rbxlx file and upload the model

Troubleshooting: Pasted the wrong ids or accidentally pressed enter without input? You can edit the ids in the roblox_asset_ids.json and rerun roblox_model_export.py to update the rblx file
