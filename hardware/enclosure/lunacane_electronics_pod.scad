// LunaCane electronics pod concept model.
// Open in OpenSCAD, tune parameters, then export STL for 3D printing.

$fn = 64;

// Cane and enclosure parameters, in millimeters.
cane_diameter = 22;
rubber_pad_thickness = 1.5;

pod_length = 118;
pod_width = 48;
pod_height = 27;
wall = 2.2;
corner_radius = 5;

battery_length = 78;
battery_width = 28;
battery_height = 23;

clamp_width = 18;
clamp_gap = 7;
clamp_wall = 4;

screw_diameter = 2.6;
screw_head_diameter = 5.2;
screw_offset_x = 48;
screw_offset_y = 16;

button_diameter = 12;
mic_hole_diameter = 3;
speaker_hole_diameter = 3;

module rounded_box(size, r) {
    hull() {
        for (x = [-size[0] / 2 + r, size[0] / 2 - r])
            for (y = [-size[1] / 2 + r, size[1] / 2 - r])
                for (z = [-size[2] / 2 + r, size[2] / 2 - r])
                    translate([x, y, z]) sphere(r = r);
    }
}

module screw_holes(z_offset = 0) {
    for (x = [-screw_offset_x, screw_offset_x])
        for (y = [-screw_offset_y, screw_offset_y])
            translate([x, y, z_offset])
                cylinder(h = pod_height + 8, d = screw_diameter, center = true);
}

module screw_posts() {
    for (x = [-screw_offset_x, screw_offset_x])
        for (y = [-screw_offset_y, screw_offset_y])
            translate([x, y, -pod_height / 2 + wall + 4])
                difference() {
                    cylinder(h = 8, d = 7, center = true);
                    cylinder(h = 10, d = screw_diameter, center = true);
                }
}

module speaker_grille() {
    for (x = [-12, -6, 0, 6, 12])
        for (z = [-5, 0, 5])
            translate([x, -pod_width / 2 - 1, z])
                rotate([90, 0, 0])
                    cylinder(h = 5, d = speaker_hole_diameter, center = true);
}

module top_io_cutouts() {
    translate([-34, 0, pod_height / 2])
        cylinder(h = 8, d = button_diameter, center = true);

    translate([-12, 0, pod_height / 2])
        cylinder(h = 8, d = mic_hole_diameter, center = true);

    translate([5, 0, pod_height / 2])
        cylinder(h = 8, d = mic_hole_diameter, center = true);
}

module usb_service_cutout() {
    translate([pod_length / 2 + 1, 0, 1])
        cube([5, 14, 9], center = true);
}

module main_pod_shell() {
    difference() {
        rounded_box([pod_length, pod_width, pod_height], corner_radius);
        translate([0, 0, wall])
            rounded_box(
                [pod_length - 2 * wall, pod_width - 2 * wall, pod_height],
                max(1, corner_radius - wall)
            );
        screw_holes();
        top_io_cutouts();
        speaker_grille();
        usb_service_cutout();
    }
    screw_posts();
}

module battery_sled() {
    translate([0, pod_width / 2 + battery_width / 2 + 5, 0])
        difference() {
            rounded_box([battery_length, battery_width, battery_height], 4);
            translate([0, 0, wall])
                rounded_box([battery_length - 2 * wall, battery_width - 2 * wall, battery_height], 2);
            translate([battery_length / 2 + 1, 0, 2])
                cube([5, 12, 8], center = true);
        }
}

module cane_clamp() {
    outer_d = cane_diameter + 2 * rubber_pad_thickness + 2 * clamp_wall;
    inner_d = cane_diameter + 2 * rubber_pad_thickness;

    translate([0, -pod_width / 2 - outer_d / 2 + 2, -pod_height / 2 + 3])
        rotate([90, 0, 0])
            difference() {
                cylinder(h = clamp_width, d = outer_d, center = true);
                cylinder(h = clamp_width + 2, d = inner_d, center = true);
                translate([0, outer_d / 2, 0])
                    cube([clamp_gap, outer_d, clamp_width + 4], center = true);
            }
}

module internal_module_placeholders() {
    color("green")
        translate([-28, 0, -pod_height / 2 + wall + 1])
            cube([50, 28, 2], center = true); // ESP32-S3 board area

    color("blue")
        translate([24, 12, -pod_height / 2 + wall + 1])
            cube([20, 16, 2], center = true); // BMI270 board area

    color("orange")
        translate([26, -11, -pod_height / 2 + wall + 1])
            cube([24, 18, 2], center = true); // I2S amp area
}

module lunacane_pod(show_placeholders = true) {
    color("lightgray") main_pod_shell();
    color("silver") battery_sled();
    color("gray") cane_clamp();
    if (show_placeholders) internal_module_placeholders();
}

lunacane_pod(true);
