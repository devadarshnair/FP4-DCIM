// Extra simulation root: back-annotate cell delays (Icarus: compile with -gspecify).
module sdf_annotate;
  initial $sdf_annotate(`SDF_FILE, `SDF_TOP);
endmodule
