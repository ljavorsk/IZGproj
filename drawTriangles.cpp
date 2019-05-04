#include <student/gpu.h>

#include<iostream>
#include<stdio.h>
#include<assert.h>
#include<string.h>
#include<math.h>

#define MIN(a, b)       (((a) < (b)) ? (a) : (b))
#define MAX(a, b)       (((a) > (b)) ? (a) : (b))

typedef struct {
  GPUOutVertex vertices[3];
} Triangle_t;

bool isPointInsideViewFrustum(Vec4 const*const p);

void triangleViewportTransform(GPU *const gpu, Triangle_t *triangle) {
    for (int i = 0; i < 3; ++i) {
        Vec4 *vertexPos = &triangle->vertices[i].gl_Position;
        vertexPos->data[0] = (vertexPos->data[0] * 0.5f + 0.5f) * gpu->framebuffer.width; // x coord
        vertexPos->data[1] = (vertexPos->data[1] * 0.5f + 0.5f) * gpu->framebuffer.height; // y coord
    }
}
 
void trianglePerspectiveDivision(Triangle_t *triangle){
  for (int i = 0; i < 3; ++i) {
    for(int a=0;a<3;++a)
      triangle->vertices[i].gl_Position.data[a] = 
        triangle->vertices[i].gl_Position.data[a] / triangle->vertices[i].gl_Position.data[3];
  }
}

Vec4 computeFragPosition(Vec4 const&p,uint32_t width,uint32_t height);

void copyVertexAttribute(GPU const*const gpu,GPUAttribute*const att,GPUVertexPullerHead const*const head,uint64_t vertexId);

void lobotomized_vertexPullerTriangle(GPUInVertex*const inVertex,GPUVertexPuller const*const vao,GPU const*const gpu,uint32_t vertexSHaderInvocation){
  uint32_t vertexId = vertexSHaderInvocation;
  inVertex->gl_VertexID = vertexId;
  if(gpu_isBuffer(gpu, vao->indices.bufferId)){
    const GPUBuffer *tmp_buff = gpu_getBuffer(gpu, vao->indices.bufferId);
    if(vao->indices.type == UINT8) 
      inVertex->gl_VertexID = ((uint8_t *)tmp_buff->data)[vertexId];
    if(vao->indices.type == UINT16) 
      inVertex->gl_VertexID = ((uint16_t *)tmp_buff->data)[vertexId];
    if(vao->indices.type == UINT32) 
      inVertex->gl_VertexID = ((uint32_t *)tmp_buff->data)[vertexId];
  }

  for(int i=0; i < MAX_ATTRIBUTES; i ++){
    copyVertexAttribute(gpu,inVertex->attributes+i,vao->heads+i,vertexId);
  }
}

/* Skopírované z cvičenia na rasterizáciu trojuholníkov (Pinedov algoritmus)
 * Upravené špecificky pre tento projekt 
 */
void triangleRasterization(GPUFragmentShaderData*const fragment, GPU*const gpu, GPUProgram const* prg, Triangle_t triangle) {
  Vec4 v1 = triangle.vertices[0].gl_Position;
  Vec4 v2 = triangle.vertices[1].gl_Position;
  Vec4 v3 = triangle.vertices[2].gl_Position;

	int min_x = MIN(MIN(v1.data[0], v2.data[0]), v3.data[0]);
	int min_y = MIN(MIN(v1.data[1], v2.data[1]), v3.data[1]);

	int max_x = MAX(MAX(v1.data[0], v2.data[0]), v3.data[0]);
	int max_y = MAX(MAX(v1.data[1], v2.data[1]), v3.data[1]);

  min_x = MIN(MAX(min_x, 0), gpu->framebuffer.width - 1);
  max_x = MIN(MAX(max_x, 0), gpu->framebuffer.width - 1);
  min_y = MIN(MAX(min_y, 0), gpu->framebuffer.height - 1);
  max_y = MIN(MAX(max_y, 0), gpu->framebuffer.height - 1);

	int delta_x1 = v2.data[0] - v1.data[0];
	int delta_x2 = v3.data[0] - v2.data[0];
	int delta_x3 = v1.data[0] - v3.data[0];

	int delta_y1 = v2.data[1] - v1.data[1];
	int delta_y2 = v3.data[1] - v2.data[1];
	int delta_y3 = v1.data[1] - v3.data[1];

	int edge_f1 = (min_y - v1.data[1] + 0.5)* delta_x1 - (min_x - v1.data[0] + 0.5)* delta_y1;
	int edge_f2 = (min_y - v2.data[1] + 0.5)* delta_x2 - (min_x - v2.data[0] + 0.5)* delta_y2;
	int edge_f3 = (min_y - v3.data[1] + 0.5)* delta_x3 - (min_x - v3.data[0] + 0.5)* delta_y3;

	for (int y = min_y; y <= max_y; y++) {

		bool even = (y - min_y) % 2 == 0;

		for (int x = ((even) ? min_x : max_x); (even) ? (x <= max_x) : (x >= min_x); x+= (even) ? 1 : -1){
			
			if (edge_f1 >= 0 && edge_f2 >= 0 && edge_f3 >= 0) {
        GPUFragmentShaderData fd;
        fd.uniforms = &prg->uniforms;
        Vec4 coord = { (float)(x+0.5), (float)(y+0.5),0,1};
        fd.inFragment.gl_FragCoord = coord;
        //inFragment.gl_FragCoord = coord; // x+0.5, y + 0.5, z zinterpolujes
        // interpolace atributu
        prg->fragmentShader(&fd);

       //lobotomized_perFragmentOperation(&fd.outFragment,gpu,ndc);
			}

			if ( !((even && x == max_x) || (!even && x == min_x) ) ) {

				edge_f1 += (even) ? -delta_y1 : delta_y1;
				edge_f2 += (even) ? -delta_y2 : delta_y2;
				edge_f3 += (even) ? -delta_y3 : delta_y3;
			}
			
		}
		edge_f1 += delta_x1;
		edge_f2 += delta_x2;
		edge_f3 += delta_x3;

	}
}

void lobotomized_perFragmentOperation(GPUOutFragment const*const outFragment,GPU*const gpu,Vec4 ndc);

/**
 * @brief This function draws points
 * 
 * @param gpu gpu
 * @param nofVertices number of vertices
 */
void gpu_drawTriangles(GPU*const gpu,uint32_t nofVertices){
  GPUProgram      const* prg = gpu_getActiveProgram(gpu);
  GPUVertexPuller const* vao = gpu_getActivePuller (gpu);

  GPUVertexShaderData   vd;
  GPUFragmentShaderData fd;

  vd.uniforms = &prg->uniforms;
  Triangle_t triangle;
  int count = 0;

  for(uint32_t v=0;v<nofVertices;++v){

    lobotomized_vertexPullerTriangle(&vd.inVertex,vao,gpu,v);

    prg->vertexShader(&vd);

    triangle.vertices[count] = vd.outVertex;
    count++;

    if (count == 3) {
      trianglePerspectiveDivision(&triangle);
      triangleViewportTransform(gpu, &triangle);
      triangleRasterization(&fd, gpu, prg, triangle);
      count = 0;
    }
  }
}
