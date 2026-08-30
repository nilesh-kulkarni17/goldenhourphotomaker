# Golden Hour - Realistic Sunset Lighting for Low Light Portaits

**Golden Hour** is a privacy-first, browser-based sunlight editor that transforms ordinary photographs into warm, directional golden-hour scenes without relying on cloud processing, external APIs, or desktop editing software.

Drop any photo into the editor and Golden Hour performs local scene analysis to understand both **depth and semantic content**. It identifies elements such as the sky, water, people, ground, buildings, and foliage, then uses that information to apply directional sunlight selectively. Instead of treating the entire image as a flat surface, the lighting pass adapts to different scene regions, helping preserve the visual structure of the original photograph.

For example, open water remains visually distinct instead of becoming a solid shadow region, while people, buildings, and vegetation can receive directional illumination, rim lighting, and contact shadows.

## Fully Local AI Processing

All inference runs directly on the user's machine inside the browser. **Depth Anything V2 Small** is used for monocular depth estimation, while a **SegFormer model fine-tuned on ADE20K** provides semantic segmentation. The models and the Transformers.js runtime are included in the project's `vendor` folder, allowing the application to operate without sending photographs to a remote server.

There is **no backend, account, telemetry, or image upload**.

## Aim the Sun

A compass-style sun control lets you position the virtual light source by adjusting its direction and elevation. Move the sun toward the edge for lower, more dramatic raking light or toward the center for a higher, softer lighting direction.

Additional controls adjust lighting intensity and shadow strength. Built-in image controls provide brightness, contrast, hue, saturation, and sepia adjustments before the sunlight pass is applied.

## Manual Brush Controls

Automatic lighting can be refined using an add-and-subtract brush. Users can paint additional sunlight into selected areas or restore the original image where lighting should be reduced.

Brushes support solid, feathered, and soft edges, with circle, square, rectangle, triangle, and line shapes. Size and opacity are fully adjustable.

## Built for Creative Work

Golden Hour is designed for **photographers, designers, creators, and visual experimentation**. It is a static HTML, CSS, and JavaScript application that can be served locally or over HTTPS. After the page loads its models, processing happens entirely within the browser.

Your photos stay in the tab.
