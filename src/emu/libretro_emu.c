#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include <errno.h>
#include <dlfcn.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#include "libretro.h"
#include "libretro_emu.h"

static double g_nominal_fps = 60.0;
static unsigned long long g_run_count = 0;

static int g_hold[RETRO_DEVICE_ID_JOYPAD_R3+1] = {0};
static int g_press_hold_frames = 2;
static int g_press_total_frames = 1;
static int g_video_enabled = 1;
static int g_frameskip = 0;

static uint8_t *g_last_rgb = NULL;
static unsigned g_last_w = 0, g_last_h = 0;

static enum retro_pixel_format g_retro_pixfmt = RETRO_PIXEL_FORMAT_XRGB8888;
static struct retro_memory_map g_memmap = {0};
static struct retro_memory_descriptor *g_memmap_desc = NULL;
static const uint8_t *g_sysram = NULL;
static size_t g_sysram_size = 0;

// When non-zero we suppress libretro/core output (log callback and
// temporarily redirect stdout/stderr around core-run calls).
static int g_suppress_core_output = 0;
static int g_core_stdout_saved = -1;
static int g_core_stderr_saved = -1;
static int g_core_null_fd = -1;
static int g_core_output_redirected = 0;

static unsigned g_joy[RETRO_DEVICE_ID_JOYPAD_R3+1] = { 0 };

#define load_sym(V, S) do {\
	if (!((*(void**)&V) = dlsym(g_retro.handle, #S))) \
		die("Failed to load symbol '" #S "'': %s", dlerror()); \
	} while (0)
#define load_retro_sym(S) load_sym(g_retro.S, S)

static void core_load(const char *sofile);
static void core_load_game(const char *filename);
static void core_unload(void);
static void core_apply_output_state(void);
static void refresh_memory_cache(void);

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
	if (g_suppress_core_output) return;
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
	case RETRO_ENVIRONMENT_GET_LOG_INTERFACE: {
		struct retro_log_callback *cb = (struct retro_log_callback *)data;
		cb->log = core_log;
		return true;
	}
	case RETRO_ENVIRONMENT_GET_CAN_DUPE:
		bval = (bool*)data;
		*bval = true;
		return true;
	case RETRO_ENVIRONMENT_SET_PIXEL_FORMAT: {
		const enum retro_pixel_format *fmt = (const enum retro_pixel_format *)data;
		if (*fmt > RETRO_PIXEL_FORMAT_RGB565) return false;
		g_retro_pixfmt = *fmt;
		return true;
	}
	case RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY:
	case RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY:
		*(const char **)data = "."; 
		return true;
	case RETRO_ENVIRONMENT_GET_VARIABLE: {
		struct retro_variable *var = (struct retro_variable *)data;
		if (var && var->key) {
			if (strcmp(var->key, "mgba_frameskip") == 0) {
				static char fs_str[16];
				snprintf(fs_str, sizeof(fs_str), "%d", g_frameskip);
				var->value = fs_str;
				return true;
			}
			if (strcmp(var->key, "mgba_idle_optimization") == 0) {
				var->value = (g_frameskip > 0) ? "Detect and Remove" : "Don't Remove";
				return true;
			}
		}
		return false;
	}
	case RETRO_ENVIRONMENT_SET_MEMORY_MAPS: {
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
	}
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
	if (!g_video_enabled) return;
	convert_to_rgb(data, width, height, pitch, g_retro_pixfmt);
}

static void core_input_poll(void) {
	int ids[] = {RETRO_DEVICE_ID_JOYPAD_A,RETRO_DEVICE_ID_JOYPAD_B,RETRO_DEVICE_ID_JOYPAD_UP,RETRO_DEVICE_ID_JOYPAD_DOWN,RETRO_DEVICE_ID_JOYPAD_LEFT,RETRO_DEVICE_ID_JOYPAD_RIGHT, RETRO_DEVICE_ID_JOYPAD_START};
	for (unsigned k=0;k<sizeof(ids)/sizeof(ids[0]);++k) {
		int id = ids[k];
		g_joy[id] = (g_hold[id] > 0) ? 1 : 0;
		if (g_hold[id] > 0) {
			g_hold[id]--;
		}
	}
}

static int16_t core_input_state(unsigned port, unsigned device, unsigned index, unsigned id) {
	if (port || index || device != RETRO_DEVICE_JOYPAD)
		return 0;
	return (int16_t)g_joy[id];
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

	/* Respect LIBRETRO_SILENT=1 or =true to silence core output by default */
	const char *silent_env = getenv("LIBRETRO_SILENT");
	if (silent_env) {
		if (silent_env[0] == '1' || strcasecmp(silent_env, "true") == 0)
			g_suppress_core_output = 1;
	}

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
	core_apply_output_state();

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
	info.size = (size_t)ftell(file);
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

	if (g_core_output_redirected) {
		fflush(stdout);
		fflush(stderr);
		if (g_core_stdout_saved >= 0) {
			dup2(g_core_stdout_saved, STDOUT_FILENO);
			close(g_core_stdout_saved);
		}
		if (g_core_stderr_saved >= 0) {
			dup2(g_core_stderr_saved, STDERR_FILENO);
			close(g_core_stderr_saved);
		}
		if (g_core_null_fd >= 0) {
			close(g_core_null_fd);
		}
		g_core_stdout_saved = -1;
		g_core_stderr_saved = -1;
		g_core_null_fd = -1;
		g_core_output_redirected = 0;
	}

	if (g_retro.handle)
		dlclose(g_retro.handle);
}

static void core_apply_output_state(void) {
	if (g_suppress_core_output && !g_core_output_redirected) {
		fflush(stdout);
		fflush(stderr);
		g_core_stdout_saved = dup(STDOUT_FILENO);
		g_core_stderr_saved = dup(STDERR_FILENO);
		g_core_null_fd = open("/dev/null", O_WRONLY);
		if (g_core_null_fd >= 0) {
			dup2(g_core_null_fd, STDOUT_FILENO);
			dup2(g_core_null_fd, STDERR_FILENO);
		}
		g_core_output_redirected = 1;
	} else if (!g_suppress_core_output && g_core_output_redirected) {
		fflush(stdout);
		fflush(stderr);
		if (g_core_stdout_saved >= 0) {
			dup2(g_core_stdout_saved, STDOUT_FILENO);
			close(g_core_stdout_saved);
		}
		if (g_core_stderr_saved >= 0) {
			dup2(g_core_stderr_saved, STDERR_FILENO);
			close(g_core_stderr_saved);
		}
		if (g_core_null_fd >= 0) {
			close(g_core_null_fd);
		}
		g_core_stdout_saved = -1;
		g_core_stderr_saved = -1;
		g_core_null_fd = -1;
		g_core_output_redirected = 0;
	}
}

static void refresh_memory_cache(void) {
	g_sysram = (const uint8_t *)g_retro.retro_get_memory_data(RETRO_MEMORY_SYSTEM_RAM);
	g_sysram_size = g_retro.retro_get_memory_size(RETRO_MEMORY_SYSTEM_RAM);
}

static inline int sysram_read_offset(uint32_t addr, size_t nbytes, size_t *offset) {
	const uint32_t bases[2] = { 0x02000000u, 0x03000000u };
	for (int b = 0; b < 2; ++b) {
		uint32_t base = bases[b];
		if (addr >= base) {
			size_t off = (size_t)(addr - base);
			if (off + nbytes <= g_sysram_size) {
				*offset = off;
				return 1;
			}
		}
	}
	return 0;
}

static inline uint8_t fast_read8(uint32_t addr, int *ok) {
	size_t off = 0;
	if (g_sysram && g_sysram_size && sysram_read_offset(addr, 1, &off)) {
		if (ok) *ok = 1;
		return g_sysram[off];
	}
	if (ok) *ok = 0;
	return 0;
}

static inline uint16_t fast_read16(uint32_t addr, int *ok) {
	size_t off = 0;
	if (g_sysram && g_sysram_size && sysram_read_offset(addr, 2, &off)) {
		uint16_t value;
		memcpy(&value, g_sysram + off, sizeof(value));
		if (ok) *ok = 1;
		return value;
	}
	if (ok) *ok = 0;
	return 0;
}

static inline uint32_t fast_read32(uint32_t addr, int *ok) {
	size_t off = 0;
	if (g_sysram && g_sysram_size && sysram_read_offset(addr, 4, &off)) {
		uint32_t value;
		memcpy(&value, g_sysram + off, sizeof(value));
		if (ok) *ok = 1;
		return value;
	}
	if (ok) *ok = 0;
	return 0;
}

static inline uint8_t read8(uint32_t addr) {
	int ok = 0;
	uint8_t value = fast_read8(addr, &ok);
	if (ok) return value;
	long v = gba_ram(addr, 1);
	return (v < 0) ? 0 : (uint8_t)(v & 0xFF);
}

static inline uint16_t read16(uint32_t addr) {
	int ok = 0;
	uint16_t value = fast_read16(addr, &ok);
	if (ok) return value;
	long v = gba_ram(addr, 2);
	return (v < 0) ? 0 : (uint16_t)(v & 0xFFFF);
}

static inline uint32_t read32(uint32_t addr) {
	int ok = 0;
	uint32_t value = fast_read32(addr, &ok);
	if (ok) return value;
	long v = gba_ram(addr, 4);
	return (v < 0) ? 0 : (uint32_t)(v & 0xFFFFFFFF);
}

// Wrapper to call the core's retro_run while optionally silencing
// its stdout/stderr output. Increments global run counter like callers
// previously did.
static void core_run_silent_wrapper(void) {
	if (!g_retro.retro_run) return;

	g_retro.retro_run();

	++g_run_count;
}

// Public API to toggle silencing of the core. Also allow enabling via
// the LIBRETRO_SILENT environment variable when loading a core.
void gba_set_core_silent(int silent) {
	g_suppress_core_output = silent ? 1 : 0;
	core_apply_output_state();
}

void gba_set_press_timing(int hold_frames, int total_frames) {
	if (hold_frames < 1) hold_frames = 1;
	if (total_frames < 1) total_frames = 1;
	if (total_frames < hold_frames) total_frames = hold_frames;
	g_press_hold_frames = hold_frames;
	g_press_total_frames = total_frames;
}

void gba_set_video_enabled(int enabled) {
	g_video_enabled = enabled ? 1 : 0;
}

void gba_set_frameskip(int frameskip) {
	if (frameskip < 0) frameskip = 0;
	g_frameskip = frameskip;
}

long gba_ram(uint32_t addr, size_t nbytes) {
	if (nbytes == 0) return 0;
	if (!g_sysram || g_sysram_size == 0) {
		refresh_memory_cache();
	}
	if (!g_sysram || g_sysram_size == 0) return -1;
	const uint8_t *base = g_sysram;
	const uint32_t bases[2] = { 0x02000000u, 0x03000000u };
	for (int b = 0; b < 2; ++b) {
		uint32_t A = bases[b];
		if (addr >= A) {
			size_t off = (size_t)(addr - A);
			if (off + nbytes <= g_sysram_size) {
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

	g_hold[id] = g_press_hold_frames;
	int orig_video = g_video_enabled;
	for (int i = 0; i < g_press_total_frames; ++i) {
		if (orig_video) {
			g_video_enabled = (i == g_press_total_frames - 1) ? 1 : 0;
		}
		core_run_silent_wrapper();
	}
	g_video_enabled = orig_video;

	return 0;
}

int gba_reset(const char* savestate) {
	memset(g_hold, 0, sizeof(g_hold));
	memset(g_joy,  0, sizeof(g_joy));

	FILE *fd = fopen(savestate, "rb");
	if (!fd)
		die("Failed to find savestate file '%s'", savestate);

	void *saveblob = malloc(g_retro.retro_serialize_size());
	size_t rdb = fread(saveblob, 1, g_retro.retro_serialize_size(), fd);

	if (!g_retro.retro_unserialize(saveblob, rdb))
		die("Failed to load savestate, core returned error");
	refresh_memory_cache();
	fclose(fd);
	free(saveblob);

	return 0;
}

void gba_run(int frames) {
	int orig_video = g_video_enabled;
	for (int i = 0; i < frames; ++i) {
		if (orig_video) {
			g_video_enabled = (i == frames - 1) ? 1 : 0;
		}
		core_run_silent_wrapper();
	}
	g_video_enabled = orig_video;
}

int gba_create(const char* core_so, const char* rom_path) {
	if (!core_so || !rom_path) return -1;

	core_load(core_so);
	core_load_game(rom_path);

	for (int i = 0; i < 5; ++i) {
		core_run_silent_wrapper();
	}

	return 0;
}

void gba_savestate(const char* save_path) {
	FILE *fd = fopen(save_path, "wb");
	if (!fd) {
		fprintf(stderr, "Could not write savestate dump to '%s'", save_path);
	} else {
		size_t savsz = g_retro.retro_serialize_size();
		void *saveblob = malloc(savsz);
		if (!g_retro.retro_serialize(saveblob, savsz)) {
			fprintf(stderr, "Could not generate savestate, core returned error");
		} else {
			fwrite(saveblob, 1, savsz, fd);
		}
		free(saveblob);
		fclose(fd);
	}
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
	uint8_t hdr[14] = {'B','M',0,0,0,0,0,0,0,0,54,0,0,0};
	hdr[2]= (uint8_t)(file_size & 0xFF); hdr[3]=(uint8_t)((file_size>>8)&0xFF);
	hdr[4]= (uint8_t)((file_size>>16)&0xFF); hdr[5]=(uint8_t)((file_size>>24)&0xFF);
	fwrite(hdr,1,14,f);
	uint8_t dib[40] = {0};
	dib[0]=40;
	dib[4]= (uint8_t)(w & 0xFF); dib[5]=(uint8_t)((w>>8)&0xFF); dib[6]=(uint8_t)((w>>16)&0xFF); dib[7]=(uint8_t)((w>>24)&0xFF);
	dib[8]= (uint8_t)(h & 0xFF); dib[9]=(uint8_t)((h>>8)&0xFF); dib[10]=(uint8_t)((h>>16)&0xFF); dib[11]=(uint8_t)((h>>24)&0xFF);
	dib[12]=1; dib[14]=24;
	dib[20]= (uint8_t)(data_size & 0xFF); dib[21]=(uint8_t)((data_size>>8)&0xFF);
	dib[22]= (uint8_t)((data_size>>16)&0xFF); dib[23]=(uint8_t)((data_size>>24)&0xFF);
	fwrite(dib,1,40,f);
	uint8_t padbuf[3] = {0,0,0};
	for (int y=(int)h-1; y>=0; --y) {
		const uint8_t *row = rgb + (size_t)y*w*3;
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

// Fonction pour déterminer le comportement d'une tile
static int eval_tile_property(int type, uint16_t tile_id, uint8_t col, int mx, int my, uint32_t events, uint8_t count, uint32_t warps, uint32_t attrs) {
    if (type == 0) { // Collision
        return (col != 0) ? 1 : 0;
    }
    
    if (type == 1) { // Interaction
        if (tile_id == 655) return 1; // Clock
        
        // Check warps
        if (events >= 0x08000000) {
			for (unsigned w = 0; warps >= 0x08000000 && w < count; w++) {
				if ((int16_t)read16(warps + (uint32_t)w * 8u) == mx && (int16_t)read16(warps + (uint32_t)w * 8u + 2u) == my) return 1;
			}
        }
        return 0;
    }
    
    if (type == 2) { // Grass
        // On utilise le tileset primaire ou secondaire selon l'ID de la tile
		if (attrs >= 0x08000000) {
			uint32_t tile_off = (uint32_t)(tile_id % 512) * 2u;
			uint16_t behavior16 = read16(attrs + tile_off);
			uint8_t behavior = (uint8_t)(behavior16 & 0xFFu);
			return (behavior == 0x02 || behavior == 0x03) ? 1 : 0;
		}
    }
    
    return 0;
}

// Fonction pour générer une map de "comportement" (collision, interaction, herbe) autour du joueur avec des 0 et des 1 seulement
void gba_behavior_map(int type, int state[INPUT_SIZE], int player_x, int player_y) {
    uint32_t layout = read32(0x02037318);
	uint32_t map_w = read32(layout + 0x00);
	uint32_t map_h = read32(layout + 0x04);
    uint32_t map_data = read32(layout + 0x0C);
    uint32_t ts_pri = read32(layout + 0x10);
    uint32_t ts_sec = read32(layout + 0x14);
	uint32_t events = 0;
	uint8_t count = 0;
	uint32_t warps = 0;
	uint32_t attrs = 0;

	if (type == 1) {
		events = read32(0x0203731C);
		if (events >= 0x08000000) {
			count = read8(events + 0x01);
			warps = read32(events + 0x08);
		}
	} else if (type == 2) {
		uint32_t ts = ts_pri;
		attrs = ts ? read32(ts + 0x10) : 0;
		if (attrs < 0x08000000) {
			ts = ts_sec;
			attrs = ts ? read32(ts + 0x10) : 0;
		}
	}

    int px = player_x - 7;
	int py = player_y - 7;
    
	// On construit une map 11x11 autour du joueur
	int behavior_out[11][11] = {0};
    for (int r = 0; r < 11; r++) {
        for (int c = 0; c < 11; c++) {
            int mx = px + (c - 5);
            int my = py + (r - 5);
            
			// Check des limites de la map
			if (mx < 0 || my < 0 || (uint32_t)mx >= map_w || (uint32_t)my >= map_h) {
                behavior_out[r][c] = (type == 0) ? 1 : 0; // Type 0 : collision
                continue;
            }
            
			uint32_t pos = (uint32_t)my * map_w + (uint32_t)mx;
			uint16_t tile = read16(map_data + pos * 2u);
			behavior_out[r][c] = eval_tile_property(type, tile & 0x03FF, (tile >> 10) & 0x03, mx, my, events, count, warps, attrs);
        }
    }

	// Copier le résultat dans l'état
	for (int i=0; i<121; i++) {
		state[24 + type*121 + i] = behavior_out[i/11][i%11];
	}
}

void gba_state(int state[INPUT_SIZE]) {
	// Organisation de l'état
	// 0 - Zone actuelle
	// 1 - Position x du joueur
	// 2 - Position y du joueur
	// 3-20 - Level, HP, max HP des 6 pokémons du joueur
	// 21-23 - Level, HP, max HP du pokémon adverse
	// 24 - 144 - Behavior map 11x11 des collisions (1 si la case est solide, 0 sinon)
	// 145 - 265 - Behavior map 11x11 des interactions (1 si la case a une interaction, 0 sinon)
	// 266 - 386 - Behavior map 11x11 des hautes herbes (1 si la case est une haute herbe, 0 sinon)

	state[0] = (int)read16(0x020322E4u); // Zone
	state[1] = (int)read16(0x02037360u); // Player X
	state[2] = (int)read16(0x02037362u); // Player Y

	const uint32_t bases[7] = { 0x02024540u, 0x020245A4u, 0x02024608u, 0x0202466Cu, 0x020246D0u, 0x02024734u, 0x02024104u };
	for (int p = 0; p < 7; p++) {
		state[3 + p*3 + 0] = (int)read8(bases[p]); 		   // Level
		state[3 + p*3 + 1] = (int)read16(bases[p] + 0x02); // HP
		state[3 + p*3 + 2] = (int)read16(bases[p] + 0x04); // Max HP
	}

	gba_behavior_map(0, state, (int)state[1], (int)state[2]); // Collision
	gba_behavior_map(1, state, (int)state[1], (int)state[2]); // Interaction
	gba_behavior_map(2, state, (int)state[1], (int)state[2]); // Herbe

	// On fusionne les maps de collision et d'interaction
	for (int i = 0; i < 121; i++) {
		int coll_idx = 24 + i;
		int inter_idx = 145 + i;
		if (state[inter_idx] == 1) {
			state[coll_idx] = -1;
			state[inter_idx] = 0;
		}
	}
}