#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include <errno.h>
#include <dlfcn.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#include "libretro.h"
#include "gba.h"

static double g_nominal_fps = 60.0;
static unsigned long long g_run_count = 0;

static int g_hold[RETRO_DEVICE_ID_JOYPAD_R3+1] = {0};

static uint8_t *g_last_rgb = NULL;
static unsigned g_last_w = 0, g_last_h = 0;

static enum retro_pixel_format g_retro_pixfmt = RETRO_PIXEL_FORMAT_XRGB8888;
static struct retro_memory_map g_memmap = (struct retro_memory_map){0};
static struct retro_memory_descriptor *g_memmap_desc = NULL;

static unsigned g_joy[RETRO_DEVICE_ID_JOYPAD_R3+1] = { 0 };

#define load_sym(V, S) do {\
	if (!((*(void**)&V) = dlsym(g_retro.handle, #S))) \
		die("Failed to load symbol '" #S "'': %s", dlerror()); \
	} while (0)
#define load_retro_sym(S) load_sym(g_retro.S, S)

static void core_load(const char *sofile);
static void core_load_game(const char *filename);
static void core_unload(void);

static struct {
	void *handle;
	bool initialized;

	void (*retro_init)(void);
	void (*retro_deinit)(void);
	unsigned (*retro_api_version)(void);
	void (*retro_get_system_info)(struct retro_system_info *info);
	void (*retro_get_system_av_info)(struct retro_system_av_info *info);
	void (*retro_set_controller_port_device)(unsigned port, unsigned device);
	void (*retro_reset)(void);
	void (*retro_run)(void);
	size_t (*retro_serialize_size)(void);
	bool (*retro_serialize)(void *data, size_t size);
	bool (*retro_unserialize)(const void *data, size_t size);
	void* (*retro_get_memory_data)(unsigned id);
	size_t (*retro_get_memory_size)(unsigned id);
	bool (*retro_load_game)(const struct retro_game_info *game);
	void (*retro_unload_game)(void);
} g_retro;

static void die(const char *fmt, ...) {
	char buffer[4096];

	va_list va;
	va_start(va, fmt);
	vsnprintf(buffer, sizeof(buffer), fmt, va);
	va_end(va);

	fputs(buffer, stderr);
	fputc('\n', stderr);
	fflush(stderr);

	exit(EXIT_FAILURE);
}


static void core_log(enum retro_log_level level, const char *fmt, ...) {
	static const char * levelstr[] = { "dbg", "inf", "wrn", "err" };
	char buffer[4096] = {0};
	va_list va;
	va_start(va, fmt);
	vsnprintf(buffer, sizeof(buffer), fmt, va);
	va_end(va);
	if (level < 0 || level > RETRO_LOG_ERROR) level = RETRO_LOG_INFO;
	fprintf(stderr, "[%s] %s", levelstr[level], buffer);
	fflush(stderr);
}


static bool core_environment(unsigned cmd, void *data) {
	bool *bval;

	switch (cmd) {
	case RETRO_ENVIRONMENT_GET_LOG_INTERFACE:
		struct retro_log_callback *cb = (struct retro_log_callback *)data; 
		cb->log = core_log; 
		return true; 
	case RETRO_ENVIRONMENT_GET_CAN_DUPE:
		bval = (bool*)data; 
		*bval = true; 
		return true;
	case RETRO_ENVIRONMENT_SET_PIXEL_FORMAT:
		const enum retro_pixel_format *fmt = (enum retro_pixel_format *)data;
		if (*fmt > RETRO_PIXEL_FORMAT_RGB565) return false; 
		g_retro_pixfmt = *fmt; 
		return true;
	case RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY:
	case RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY:
		*(const char **)data = "."; 
		return true;
	case RETRO_ENVIRONMENT_SET_MEMORY_MAPS:
		const struct retro_memory_map *m = (const struct retro_memory_map *)data;
		free(g_memmap_desc); g_memmap_desc = NULL; g_memmap.descriptors = NULL; g_memmap.num_descriptors = 0;
		if (m && m->num_descriptors && m->descriptors) {
			g_memmap_desc = (struct retro_memory_descriptor*)malloc(sizeof(*g_memmap_desc) * m->num_descriptors);
			if (g_memmap_desc) { 
				memcpy(g_memmap_desc, m->descriptors, sizeof(*g_memmap_desc) * m->num_descriptors);
				g_memmap.descriptors = g_memmap_desc; 
				g_memmap.num_descriptors = m->num_descriptors;
			}
		}
		return true;
	default: 
		return false;
	}
}

static void ensure_rgb(unsigned w, unsigned h) {
	size_t need = (size_t)w * (size_t)h * 3;
	if (!g_last_rgb || g_last_w != w || g_last_h != h) {
		free(g_last_rgb);
		g_last_rgb = (uint8_t*)malloc(need);
		g_last_w = w; 
		g_last_h = h;
	}
}

static void convert_to_rgb(const void *src_data, unsigned w, unsigned h, size_t pitch, enum retro_pixel_format fmt) {
	const uint8_t *src = (const uint8_t*)src_data;
	ensure_rgb(w,h);
	uint8_t *dst = g_last_rgb;
	if (fmt == RETRO_PIXEL_FORMAT_XRGB8888) {
		for (unsigned y=0;y<h;++y) {
			const uint32_t *s = (const uint32_t*)(src + y*pitch);
			uint8_t *d = dst + y*w*3;
			for (unsigned x=0;x<w;++x) {
				uint32_t p = s[x];
				d[3*x+0] = (p >> 16) & 0xFF; // R
				d[3*x+1] = (p >> 8) & 0xFF;  // G
				d[3*x+2] = (p) & 0xFF;       // B
			}
		}
	} else if (fmt == RETRO_PIXEL_FORMAT_RGB565) {
		for (unsigned y=0;y<h;++y) {
			const uint16_t *s = (const uint16_t*)(src + y*pitch);
			uint8_t *d = dst + y*w*3;
			for (unsigned x=0;x<w;++x) {
				uint16_t p = s[x];
				d[3*x+0] = (uint8_t)(((p >> 11) & 0x1F) * 255 / 31);
				d[3*x+1] = (uint8_t)(((p >> 5) & 0x3F) * 255 / 63);
				d[3*x+2] = (uint8_t)((p & 0x1F) * 255 / 31);
			}
		}
	} else { // 0RGB1555
		for (unsigned y=0;y<h;++y) {
			const uint16_t *s = (const uint16_t*)(src + y*pitch);
			uint8_t *d = dst + y*w*3;
			for (unsigned x=0;x<w;++x) {
				uint16_t p = s[x];
				d[3*x+0] = (uint8_t)(((p >> 10) & 0x1F) * 255 / 31);
				d[3*x+1] = (uint8_t)(((p >> 5) & 0x1F) * 255 / 31);
				d[3*x+2] = (uint8_t)((p & 0x1F) * 255 / 31);
			}
		}
	}
}

static void core_video_refresh(const void *data, unsigned width, unsigned height, size_t pitch) {
	if (!data) return; 
	convert_to_rgb(data, width, height, pitch, g_retro_pixfmt);
}

static void core_input_poll(void) {
	int ids[] = {RETRO_DEVICE_ID_JOYPAD_A,RETRO_DEVICE_ID_JOYPAD_B,RETRO_DEVICE_ID_JOYPAD_UP,RETRO_DEVICE_ID_JOYPAD_DOWN,RETRO_DEVICE_ID_JOYPAD_LEFT,RETRO_DEVICE_ID_JOYPAD_RIGHT, RETRO_DEVICE_ID_JOYPAD_START};
	for (unsigned k=0;k<sizeof(ids)/sizeof(ids[0]);++k) {
		int id = ids[k];
		g_joy[id] = (g_hold[id] > 0) ? 1 : 0;
		if (g_hold[id] > 0) g_hold[id]--;
	}
}

static int16_t core_input_state(unsigned port, unsigned device, unsigned index, unsigned id) {
	if (port || index || device != RETRO_DEVICE_JOYPAD)
		return 0;

	return g_joy[id];
}

static void core_audio_sample(int16_t left, int16_t right) { (void)left; (void)right; }

static size_t core_audio_sample_batch(const int16_t *data, size_t frames) { (void)data; return frames; }

static void core_load(const char *sofile) {
	void (*set_environment)(retro_environment_t) = NULL;
	void (*set_video_refresh)(retro_video_refresh_t) = NULL;
	void (*set_input_poll)(retro_input_poll_t) = NULL;
	void (*set_input_state)(retro_input_state_t) = NULL;
	void (*set_audio_sample)(retro_audio_sample_t) = NULL;
	void (*set_audio_sample_batch)(retro_audio_sample_batch_t) = NULL;

	memset(&g_retro, 0, sizeof(g_retro));
	g_retro.handle = dlopen(sofile, RTLD_LAZY);

	if (!g_retro.handle)
		die("Failed to load core: %s", dlerror());

	dlerror();

	load_retro_sym(retro_init);
	load_retro_sym(retro_deinit);
	load_retro_sym(retro_api_version);
	load_retro_sym(retro_get_system_info);
	load_retro_sym(retro_get_system_av_info);
	load_retro_sym(retro_set_controller_port_device);
	load_retro_sym(retro_reset);
	load_retro_sym(retro_run);
	load_retro_sym(retro_load_game);
	load_retro_sym(retro_unload_game);
	load_retro_sym(retro_serialize_size);
	load_retro_sym(retro_serialize);
	load_retro_sym(retro_unserialize);
	load_retro_sym(retro_get_memory_data);
	load_retro_sym(retro_get_memory_size);

	load_sym(set_environment, retro_set_environment);
	load_sym(set_video_refresh, retro_set_video_refresh);
	load_sym(set_input_poll, retro_set_input_poll);
	load_sym(set_input_state, retro_set_input_state);
	load_sym(set_audio_sample, retro_set_audio_sample);
	load_sym(set_audio_sample_batch, retro_set_audio_sample_batch);

	set_environment(core_environment);
	set_video_refresh(core_video_refresh);
	set_input_poll(core_input_poll);
	set_input_state(core_input_state);
	set_audio_sample(core_audio_sample);
	set_audio_sample_batch(core_audio_sample_batch);

	g_retro.retro_init();
	g_retro.initialized = true;
	if (g_retro.retro_set_controller_port_device)
		g_retro.retro_set_controller_port_device(0, RETRO_DEVICE_JOYPAD);
}

static void core_load_game(const char *filename) {
	struct retro_system_av_info av = {0};
	struct retro_system_info system = {0};
	struct retro_game_info info = {0};
	info.path = filename;
	FILE *file = fopen(filename, "rb");

	if (!file)
		goto libc_error;

	fseek(file, 0, SEEK_END);
	info.size = ftell(file);
	rewind(file);

	g_retro.retro_get_system_info(&system);

	if (!system.need_fullpath) {
		void *buf = malloc(info.size);
		if (!buf) goto libc_error;
		size_t rd = fread(buf, 1, info.size, file);
		if (rd != info.size) { 
			free(buf); 
			goto libc_error; 
		}
		info.data = buf;
	}

	if (!g_retro.retro_load_game(&info))
		die("The core failed to load the content.");

	g_retro.retro_get_system_av_info(&av);
	if (av.timing.fps > 0.0) g_nominal_fps = av.timing.fps;

	return;

libc_error:
	die("Failed to load content '%s': %s", filename, strerror(errno));
}

static void core_unload() {
	if (g_retro.initialized)
		g_retro.retro_deinit();

	if (g_retro.handle)
		dlclose(g_retro.handle);
}

long gba_ram(uint32_t addr, size_t nbytes) {
	if (nbytes == 0) return 0;
	void *sys = g_retro.retro_get_memory_data(RETRO_MEMORY_SYSTEM_RAM);
	size_t sz = g_retro.retro_get_memory_size(RETRO_MEMORY_SYSTEM_RAM);
	if (!sys || sz == 0) return -1;
	const uint8_t *base = (const uint8_t*)sys;
	const uint32_t bases[2] = { 0x02000000u, 0x03000000u };
	for (int b = 0; b < 2; ++b) {
		uint32_t A = bases[b];
		if (addr >= A) {
			size_t off = (size_t)(addr - A);
			if (off + nbytes <= sz) {
				unsigned long val = 0; const uint8_t *p = base + off;
				for (size_t i = 0; i < nbytes; ++i) val |= ((unsigned long)p[i]) << (8*i);
				return (long)val;
			}
		}
	}

	if (g_memmap.descriptors && g_memmap.num_descriptors) {
		const struct retro_memory_descriptor *d = g_memmap.descriptors;
		for (unsigned i = 0; i < g_memmap.num_descriptors; ++i) {
			if (!d[i].ptr || d[i].len == 0) continue;
			uint64_t start = (uint64_t)d[i].start;
			uint64_t end = start + (uint64_t)d[i].len;
			if ((uint64_t)addr >= start && (uint64_t)addr + nbytes <= end) {
				size_t off = (size_t)((uint64_t)addr - start) + d[i].offset;
				const uint8_t *p = (const uint8_t*)d[i].ptr + off;
				unsigned long val = 0; for (size_t k=0;k<nbytes;++k) val |= ((unsigned long)p[k]) << (8*k);
				return (long)val;
			}
		}
	}
	return -2;
}

static inline uint8_t read8(uint32_t addr) {
	long v = gba_ram(addr, 1);
	return (v < 0) ? 0 : (uint8_t)(v & 0xFF);
}
static inline uint16_t read16(uint32_t addr) {
	long v = gba_ram(addr, 2);
	return (v < 0) ? 0 : (uint16_t)(v & 0xFFFF);
}

int gba_state(int* team_out, int* enemy_out, int* pp_out, int* zone_out, int* clock_out, int bg0_out[32][32], int bg2_out[32][32]) {
	const uint32_t bases[6] = { 0x02024540u, 0x020245A4u, 0x02024608u, 0x0202466Cu, 0x020246D0u, 0x02024734u };
	for (int i = 0; i < 6; ++i) {
		uint32_t base = bases[i];
		int o = i * 8;

		int maxHP   = read16(base + 0x04);
		int HP      = read16(base + 0x02);
		int level   = read8 (base + 0x00);
		int ATK     = read16(base + 0x06);
		int DEF     = read16(base + 0x08);
		int SPEED   = read16(base + 0x0A);
		int ATK_SPE = read16(base + 0x0C);
		int DEF_SPE = read16(base + 0x0E);

		team_out[o + 0] = maxHP;
		team_out[o + 1] = HP;
		team_out[o + 2] = level;
		team_out[o + 3] = ATK;
		team_out[o + 4] = DEF;
		team_out[o + 5] = SPEED;
		team_out[o + 6] = ATK_SPE;
		team_out[o + 7] = DEF_SPE;
	}

	enemy_out[0] = read16(0x02024108u);
	enemy_out[1] = read16(0x02024104u);
	enemy_out[2] = read8 (0x02024106u);

	pp_out[0] = read8(0x020240A8u);
	pp_out[1] = read8(0x020240A9u);
	pp_out[2] = read8(0x020240AAu);
	pp_out[3] = read8(0x020240ABu);

	*zone_out  = read16(0x020322E4u);
	*clock_out = read8 (0x0203CD9Cu);

	for (int i = 0; i < 32; ++i) {
		for (int j = 0; j < 32; ++j) {
			uint32_t addr = 0x0600F800u + (uint32_t)((i * 32 + j) * 2);
			int w = read16(addr);
			bg0_out[i][j] = w;
		}
	}

	uint8_t xoff = read8(0x04000018u);
	uint8_t yoff = read8(0x0400001Au);

	int raw[32*32];
	for (int i = 0; i < 32; ++i) {
		for (int j = 0; j < 32; ++j) {
			uint32_t addr = 0x0600E000u + (uint32_t)((i * 32 + j) * 2);
			raw[i*32 + j] = read16(addr);
		}
	}
	int tx = (int)(xoff / 8);
	int ty = (int)(yoff / 8);
	for (int r = 0; r < 32; ++r) {
		for (int c = 0; c < 32; ++c) {
			int sr = (r + ty) & 31;
			int sc = (c + tx) & 31;
			bg2_out[r][c] = raw[sr*32 + sc];
		}
	}

	return 0;
}

static inline int map_button(int button_code, unsigned *out_id) {
	switch (button_code) {
		case 0: *out_id = RETRO_DEVICE_ID_JOYPAD_UP; return 0;
		case 1: *out_id = RETRO_DEVICE_ID_JOYPAD_DOWN; return 0;
		case 2: *out_id = RETRO_DEVICE_ID_JOYPAD_LEFT; return 0;
		case 3: *out_id = RETRO_DEVICE_ID_JOYPAD_RIGHT; return 0;
		case 4: *out_id = RETRO_DEVICE_ID_JOYPAD_A; return 0;
		case 5: *out_id = RETRO_DEVICE_ID_JOYPAD_B; return 0;
		case 6: *out_id = RETRO_DEVICE_ID_JOYPAD_START; return 0;
		default: return -1;
	}
}

int gba_button(int button_code) {
	unsigned id = 5;
	map_button(button_code, &id);

	g_hold[id] = 2;
	if (g_retro.retro_run) { 
		g_retro.retro_run(); 
		++g_run_count; 
	}

	return 0;
}

int gba_reset(const char* savestate) {
	FILE *fd = fopen(savestate, "rb");
	if (!fd)
		die("Failed to find savestate file '%s'", savestate);

	void *saveblob = malloc(g_retro.retro_serialize_size());
	size_t rdb = fread(saveblob, 1, g_retro.retro_serialize_size(), fd);

	if (!g_retro.retro_unserialize(saveblob, rdb))
		die("Failed to load savestate, core returned error");
	fclose(fd);
	free(saveblob);

	return 0;
}

void gba_run(int frames) {
	for (int i = 0; i < frames; ++i) {
		g_retro.retro_run();
		++g_run_count;
	}
}

int gba_create(const char* core_so, const char* rom_path) {
	if (!core_so || !rom_path) return -1;

	core_load(core_so);
	core_load_game(rom_path);

	for (int i = 0; i < 5; ++i) { 
		if (g_retro.retro_run) { 
			g_retro.retro_run(); 
			++g_run_count; 
		} 
	}

	return 0;
}

void gba_destroy() {
	core_unload(); 
}

static int write_bmp(const char *path, const uint8_t *rgb, unsigned w, unsigned h) {
	FILE *f = fopen(path, "wb");
	if (!f) return -1;
	unsigned row_stride = w*3;
	unsigned pad = (4 - (row_stride % 4)) & 3;
	unsigned data_size = (row_stride + pad) * h;
	unsigned file_size = 14 + 40 + data_size;
	// BMP header
	uint8_t hdr[14] = {'B','M',0,0,0,0,0,0,0,0,54,0,0,0};
	hdr[2]= (uint8_t)(file_size & 0xFF); hdr[3]=(uint8_t)((file_size>>8)&0xFF);
	hdr[4]= (uint8_t)((file_size>>16)&0xFF); hdr[5]=(uint8_t)((file_size>>24)&0xFF);
	fwrite(hdr,1,14,f);
	// DIB header (BITMAPINFOHEADER)
	uint8_t dib[40] = {0};
	dib[0]=40; // header size
	dib[4]= (uint8_t)(w & 0xFF); dib[5]=(uint8_t)((w>>8)&0xFF); dib[6]=(uint8_t)((w>>16)&0xFF); dib[7]=(uint8_t)((w>>24)&0xFF);
	dib[8]= (uint8_t)(h & 0xFF); dib[9]=(uint8_t)((h>>8)&0xFF); dib[10]=(uint8_t)((h>>16)&0xFF); dib[11]=(uint8_t)((h>>24)&0xFF);
	dib[12]=1; dib[14]=24; // planes=1, bpp=24
	dib[20]= (uint8_t)(data_size & 0xFF); dib[21]=(uint8_t)((data_size>>8)&0xFF);
	dib[22]= (uint8_t)((data_size>>16)&0xFF); dib[23]=(uint8_t)((data_size>>24)&0xFF);
	fwrite(dib,1,40,f);
	// pixel data bottom-up
	uint8_t padbuf[3] = {0,0,0};
	for (int y=(int)h-1; y>=0; --y) {
		const uint8_t *row = rgb + (size_t)y*w*3;
		// write as BGR
		for (unsigned x=0;x<w;++x) {
			uint8_t bgr[3] = { row[3*x+2], row[3*x+1], row[3*x+0] };
			fwrite(bgr,1,3,f);
		}
		if (pad) fwrite(padbuf,1,pad,f);
	}
	fclose(f);
	return 0;
}

void gba_screen(const char* path) {
	write_bmp(path, g_last_rgb, g_last_w, g_last_h);
}