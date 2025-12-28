import argparse
import pickle
import numpy as np
import torch
import xatlas
import trimesh
import moderngl
import cv2
from PIL import Image
from tsr.system import TSR
from multiprocessing import Pool


def simplify_mesh(mesh_result, target_faces=10000):
    """Simplify mesh to a target number of faces."""
    mesh_result = mesh_result.simplify_quadratic_decimation(target_faces)
    return mesh_result


def run_tsr(args):
    """Load the TSR model and scene codes."""
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(args.tsr_chunk_size)
    model.to(args.device)

    with open(args.input_scene_codes, "rb") as infile:
        scene_codes = pickle.load(infile)

    return {
        "model": model,
        "scene_codes": scene_codes,
    }


def run_mesh(args):
    """Load mesh from file."""
    mesh_result = trimesh.load_mesh(args.input_mesh)
    if args.simplify:
        mesh_result = simplify_mesh(mesh_result, target_faces=args.simplify_target_faces)
    return mesh_result


def run_xatlas(args, mesh_result):
    """Generate UV mapping using xatlas with optimized packing."""
    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh_result.vertices, mesh_result.faces)
    options = xatlas.PackOptions()
    options.resolution = args.texture_resolution
    options.padding = args.texture_padding
    options.bilinear = True  # Enable bilinear filtering
    atlas.generate(pack_options=options)
    vmapping, indices, uvs = atlas[0]
    return {
        "vmapping": vmapping,
        "indices": indices,
        "uvs": uvs,
    }


def run_rasterize(args, mesh_result, xatlas_result):
    """Rasterize UV atlas to generate a texture map."""
    ctx = moderngl.create_context(standalone=True)
    basic_prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec2 in_uv;
            in vec3 in_pos;
            out vec3 v_pos;
            void main() {
                v_pos = in_pos;
                gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
            }
        """,
        fragment_shader="""
            #version 330
            in vec3 v_pos;
            out vec4 o_col;
            void main() {
                o_col = vec4(v_pos, 1.0);
            }
        """,
    )
    gs_prog = ctx.program(
        vertex_shader="""...""",  # Omitted for brevity, same as your code
        geometry_shader="""...""",  # Omitted for brevity
        fragment_shader="""...""",  # Omitted for brevity
    )
    uvs = xatlas_result["uvs"].flatten().astype("f4")
    pos = mesh_result.vertices[xatlas_result["vmapping"]].flatten().astype("f4")
    indices = xatlas_result["indices"].flatten().astype("i4")
    vbo_uvs = ctx.buffer(uvs)
    vbo_pos = ctx.buffer(pos)
    ibo = ctx.buffer(indices)
    vao_content = [
        vbo_uvs.bind("in_uv", layout="2f"),
        vbo_pos.bind("in_pos", layout="3f"),
    ]
    basic_vao = ctx.vertex_array(basic_prog, vao_content, ibo)
    gs_vao = ctx.vertex_array(gs_prog, vao_content, ibo)
    fbo = ctx.framebuffer(
        color_attachments=[
            ctx.texture(
                (args.texture_resolution, args.texture_resolution), 4, dtype="f4"
            )
        ]
    )
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 0.0)
    gs_prog["u_resolution"].value = args.texture_resolution
    gs_prog["u_dilation"].value = args.texture_padding
    gs_vao.render()
    basic_vao.render()

    fbo_bytes = fbo.color_attachments[0].read()
    fbo_np = np.frombuffer(fbo_bytes, dtype="f4").reshape(
        args.texture_resolution, args.texture_resolution, 4
    )
    return fbo_np


def run_bake(args, tsr_result, rasterize_result):
    """Bake texture maps (color, normal, etc.) from the Triplane Renderer."""
    positions = torch.tensor(rasterize_result.reshape(-1, 4)[:, :-1])
    with torch.no_grad():
        queried_grid = tsr_result["model"].renderer.query_triplane(
            tsr_result["model"].decoder,
            positions,
            tsr_result["scene_codes"][0],
        )
    rgb_f = queried_grid["color"].numpy().reshape(-1, 3)
    rgba_f = np.insert(rgb_f, 3, rasterize_result.reshape(-1, 4)[:, -1], axis=1)
    rgba_f[rgba_f[:, -1] == 0.0] = [0, 0, 0, 0]
    return rgba_f.reshape(args.texture_resolution, args.texture_resolution, 4)


def denoise_texture(texture):
    """Denoise the texture using OpenCV."""
    texture_uint8 = (texture * 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoisingColored(texture_uint8, None, 10, 10, 7, 21)
    return denoised.astype(np.float32) / 255.0


def parallel_processing(args, mesh_chunks):
    """Process mesh chunks in parallel."""
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(run_mesh, mesh_chunks)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mesh", help="Path to input mesh", required=True)
    parser.add_argument("--input-scene-codes", help="Path to input scene codes", required=True)
    parser.add_argument("--output-mesh", help="Path to output mesh (.obj)", required=True)
    parser.add_argument("--output-texture", help="Path to output texture (.png)", required=True)
    parser.add_argument("--texture-resolution", help="Resolution of output texture", type=int, default=1024)
    parser.add_argument("--texture-padding", help="Extra padding on edges of UV islands", type=int, default=1)
    parser.add_argument("--tsr-chunk-size", help="Evaluation chunk size", type=int, default=8192)
    parser.add_argument("--device", help="PyTorch device", type=str, default="cpu")
    parser.add_argument("--simplify", help="Enable mesh simplification", action="store_true")
    parser.add_argument("--simplify-target-faces", help="Target number of faces after simplification", type=int, default=10000)
    parser.add_argument("--num-workers", help="Number of parallel workers", type=int, default=4)

    args = parser.parse_args()

    print("(1/6): Load TripoSR and scene codes")
    tsr_result = run_tsr(args)

    print("(2/6): Load mesh")
    mesh_result = run_mesh(args)

    print("(3/6): Generate UVs")
    xatlas_result = run_xatlas(args, mesh_result)

    print("(4/6): Rasterize UV atlas")
    rasterize_result = run_rasterize(args, mesh_result, xatlas_result)

    print("(5/6): Sample NeRF to UV atlas")
    bake_result = run_bake(args, tsr_result, rasterize_result)

    print("(6/6): Denoise and save texture")
    denoised_texture = denoise_texture(bake_result)

    print(f"Writing texture to {args.output_texture}")
    bake_img = Image.fromarray(
        (denoised_texture * 255.0).astype(np.uint8)
    ).transpose(Image.FLIP_TOP_BOTTOM)
    bake_img.save(args.output_texture)

    print(f"Writing atlased mesh to {args.output_mesh}")
    xatlas.export(
        args.output_mesh,
        mesh_result.vertices[xatlas_result["vmapping"]],
        xatlas_result["indices"],
        xatlas_result["uvs"],
        mesh_result.vertex_normals[xatlas_result["vmapping"]],
    )


if __name__ == "__main__":
    main()
