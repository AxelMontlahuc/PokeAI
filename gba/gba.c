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
static inline uint32_t read32(uint32_t addr) {
	long v = gba_ram(addr, 4);
	return (v < 0) ? 0 : (uint32_t)(v & 0xFFFFFFFF);
}

// Returns: behavior byte (0x00=walkable, 0x01=solid/wall, 0x02-0x69=various interactables)
// Get the full u32 metatile attribute for a tile
// Get behavior byte from metatile attribute
// Tileset structure (from pret/pokeemerald):
//   +0x00 isCompressed, +0x01 isSecondary
//   +0x04 tiles, +0x08 palettes, +0x0C metatiles
//   +0x10 metatileAttributes (u16 per metatile)
//   +0x14 callback
static uint8_t get_tile_behavior(uint16_t tile_id, uint32_t primary_tileset, uint32_t secondary_tileset) {
	uint32_t tileset_ptr;
	uint16_t local_id;
	
	// NUM_METATILES_IN_PRIMARY = 512 (0x200) in pokeemerald
	if (tile_id < 512) {
		tileset_ptr = primary_tileset;
		local_id = tile_id;
	} else {
		tileset_ptr = secondary_tileset;
		local_id = tile_id - 512;
	}
	
	if (tileset_ptr == 0) return 0;
	
	// metatileAttributes pointer at tileset + 0x10
	uint32_t attr_ptr = read32(tileset_ptr + 0x10);
	if (attr_ptr == 0 || attr_ptr < 0x08000000) return 0;
	
	// Each metatile attribute is u16 (2 bytes)
	uint32_t attr_addr = attr_ptr + (uint32_t)local_id * 2;
	uint16_t attr = read16(attr_addr);
	// Behavior = bits 0-7
	return attr & 0xFF;
}

// Special tile IDs that should override behavior classification
static int classify_tile_id(uint16_t tile_id) {
	switch (tile_id) {
		case 655:	 return 2;   // Clock -> interactable
		default:     return -99; // No override
	}
}

// Check if map coordinate (mx, my) is a warp destination.
// gMapHeader is at 0x02037318 in WRAM:
//   +0x00 = mapLayout ptr  (already used)
//   +0x04 = events ptr     -> MapEvents in ROM
// MapEvents:
//   +0x00 = objectEventCount (u8)
//   +0x01 = warpCount (u8)
//   +0x02 = coordEventCount (u8)
//   +0x03 = bgEventCount (u8)
//   +0x04 = objectEvents ptr (4)
//   +0x08 = warps ptr (4)     -> array of WarpEvent
// WarpEvent (8 bytes): x(s16) y(s16) elevation(u8) warpId(u8) mapNum(u8) mapGroup(u8)
static int is_warp_tile(int mx, int my) {
	uint32_t events_ptr = read32(0x0203731C);
	if (events_ptr == 0 || events_ptr < 0x08000000) return 0;

	uint8_t warp_count = read8(events_ptr + 0x01);
	if (warp_count == 0 || warp_count > 64) return 0;

	uint32_t warps_ptr = read32(events_ptr + 0x08);
	if (warps_ptr == 0 || warps_ptr < 0x08000000) return 0;

	for (int w = 0; w < (int)warp_count; w++) {
		uint32_t warp_addr = warps_ptr + (uint32_t)w * 8;
		int16_t wx = (int16_t)read16(warp_addr + 0x00);
		int16_t wy = (int16_t)read16(warp_addr + 0x02);
		if ((int)wx == mx && (int)wy == my) return 1;
	}
	return 0;
}

// Read 11x11 behavior map centered on player position
// Uses direct map layout pointer at 0x02037318
void gba_behavior_map(int behavior_out[11][11], int* player_x, int* player_y) {
	// Direct map layout pointer (not header pointer)
	uint32_t layout_ptr = read32(0x02037318);
	if (layout_ptr == 0 || layout_ptr < 0x08000000) {
		// Layout pointer usually points to ROM (0x08xxxxxx), fall back to zeros
		for (int r = 0; r < 11; r++)
			for (int c = 0; c < 11; c++)
				behavior_out[r][c] = 0;
		*player_x = 0;
		*player_y = 0;
		return;
	}
	
	// Map dimensions: width at layout+0x00, height at layout+0x04
	uint32_t map_width = read32(layout_ptr + 0x00);
	uint32_t map_height = read32(layout_ptr + 0x04);
	
	// Sanity check dimensions
	if (map_width == 0 || map_width > 1000 || map_height == 0 || map_height > 1000) {
		for (int r = 0; r < 11; r++)
			for (int c = 0; c < 11; c++)
				behavior_out[r][c] = 0;
		*player_x = 0;
		*player_y = 0;
		return;
	}
	
	// Layout: width(4), height(4), border(4), tile_data(4), ts1_header(4), ts2_header(4)
	// Skip border pointer at +0x08
	uint32_t map_data = read32(layout_ptr + 0x0C);
	uint32_t primary_tileset = read32(layout_ptr + 0x10);
	uint32_t secondary_tileset = read32(layout_ptr + 0x14);
	
	// Player X/Y map coordinates at 0x02037360/62 (Pokémon Emerald)
	uint16_t px = read16(0x02037360);
	uint16_t py = read16(0x02037362);
	*player_x = (int)px;
	*player_y = (int)py;
	
	// Pokemon maps have a border offset of 7 tiles
	// Player coords include this border, so subtract it for map data indexing
	const int BORDER = 7;
	int adj_x = (int)px - BORDER;
	int adj_y = (int)py - BORDER;
	
	// Read 11x11 grid centered on player (using adjusted coordinates)
	for (int dr = -5; dr <= 5; dr++) {
		for (int dc = -5; dc <= 5; dc++) {
			int mx = adj_x + dc;
			int my = adj_y + dr;
			int row = dr + 5;
			int col = dc + 5;
			
			// Bounds check against actual map dimensions
			if (mx < 0 || my < 0 || (uint32_t)mx >= map_width || (uint32_t)my >= map_height) {
				behavior_out[row][col] = -1;  // Out of bounds = solid
				continue;
			}
			
			// Each metatile entry in map data is 2 bytes
			// Bits 0-9: tile ID, bits 10-11: collision, bits 12-15: elevation
			uint32_t tile_addr = map_data + (uint32_t)(my * (int)map_width + mx) * 2;
			uint16_t tile_entry = read16(tile_addr);
			uint16_t tile_id = tile_entry & 0x03FF;  // bits 0-9
			uint8_t collision = (tile_entry >> 10) & 0x03;  // bits 10-11
			
			// Get behavior byte for grass detection
			uint8_t behavior = get_tile_behavior(tile_id, primary_tileset, secondary_tileset);
			
			// Check tile ID for special objects (takes highest priority)
			int id_class = classify_tile_id(tile_id);
			if (id_class != -99) {
				behavior_out[row][col] = 2;  // Special object = interactable
			} else if (is_warp_tile(mx, my)) {
				behavior_out[row][col] = 2;  // Warp/door = interactable
			} else if (behavior == 0x02 || behavior == 0x03) {
				behavior_out[row][col] = 1;  // Tall grass (0x02) / long grass (0x03)
			} else if (collision != 0) {
				behavior_out[row][col] = -1;  // Collision flag = solid
			} else {
				behavior_out[row][col] = 0;  // Walkable
			}
		}
	}
}

int gba_state(int* team_out, int* enemy_out, int* pp_out, int* zone_out, int* clock_out, int behavior_out[11][11], int* player_x, int* player_y) {
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

	gba_behavior_map(behavior_out, player_x, player_y);

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
	/* Clear any pending button holds so the new episode starts clean. */
	memset(g_hold, 0, sizeof(g_hold));
	memset(g_joy,  0, sizeof(g_joy));

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