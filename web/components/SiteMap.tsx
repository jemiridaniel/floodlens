"use client";

import { useEffect, useRef } from "react";
import maplibregl, {
  type GeoJSONSource,
  type Map as MapLibreMap,
  type Marker,
} from "maplibre-gl";
import type { Feature, FeatureCollection } from "geojson";
import "maplibre-gl/dist/maplibre-gl.css";

/** OpenStreetMap raster tiles. Attribution is required by the ODbL and is set on the
 *  source, so MapLibre always renders it; `customAttribution` is not used because it can
 *  be collapsed away on narrow screens. */
const OSM_STYLE = {
  version: 8 as const,
  sources: {
    osm: {
      type: "raster" as const,
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      maxzoom: 19,
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    },
  },
  layers: [{ id: "osm", type: "raster" as const, source: "osm" }],
};

/** A ring of `radiusM` around the pin, as a GeoJSON polygon. MapLibre has no metre-radius
 *  circle, and a pixel radius would change meaning as the officer zooms. */
function ring(lat: number, lon: number, radiusM: number, points = 96): Feature {
  const latPerM = 1 / 110_574;
  const lonPerM = 1 / (111_320 * Math.cos((lat * Math.PI) / 180));
  const coordinates = Array.from({ length: points + 1 }, (_, i) => {
    const angle = (i / points) * 2 * Math.PI;
    return [lon + radiusM * lonPerM * Math.cos(angle), lat + radiusM * latPerM * Math.sin(angle)];
  });
  return { type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [coordinates] } };
}

function rings(lat: number, lon: number): FeatureCollection {
  return {
    type: "FeatureCollection",
    features: [
      { ...ring(lat, lon, 250), properties: { radius: 250 } },
      { ...ring(lat, lon, 500), properties: { radius: 500 } },
    ],
  };
}

export function SiteMap({
  lat,
  lon,
  onPick,
  className = "",
}: {
  lat: number;
  lon: number;
  onPick?: (lat: number, lon: number) => void;
  className?: string;
}) {
  const holder = useRef<HTMLDivElement>(null);
  const map = useRef<MapLibreMap | null>(null);
  const marker = useRef<Marker | null>(null);
  const pick = useRef(onPick);
  pick.current = onPick;

  useEffect(() => {
    if (!holder.current || map.current) return;
    const m = new maplibregl.Map({
      container: holder.current,
      style: OSM_STYLE,
      center: [lon, lat],
      zoom: 15,
      attributionControl: { compact: false },
    });
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    m.on("load", () => {
      m.addSource("rings", { type: "geojson", data: rings(lat, lon) });
      m.addLayer({
        id: "ring-fill",
        type: "fill",
        source: "rings",
        paint: { "fill-color": "#1D5FA8", "fill-opacity": 0.07 },
      });
      m.addLayer({
        id: "ring-line",
        type: "line",
        source: "rings",
        paint: { "line-color": "#1D5FA8", "line-width": 1.5, "line-dasharray": [2, 2] },
      });
      m.addLayer({
        id: "ring-label",
        type: "symbol",
        source: "rings",
        layout: {
          "text-field": ["concat", ["to-string", ["get", "radius"]], " m"],
          "text-size": 11,
          "symbol-placement": "line",
        },
        paint: { "text-color": "#1D5FA8", "text-halo-color": "#FFFFFF", "text-halo-width": 1.5 },
      });
    });
    if (pick.current) {
      m.on("click", (e) => pick.current?.(e.lngLat.lat, e.lngLat.lng));
      m.getCanvas().style.cursor = "crosshair";
    }
    marker.current = new maplibregl.Marker({ color: "#1D5FA8" }).setLngLat([lon, lat]).addTo(m);
    map.current = m;
    return () => {
      m.remove();
      map.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const m = map.current;
    if (!m) return;
    marker.current?.setLngLat([lon, lat]);
    const source = m.getSource("rings") as GeoJSONSource | undefined;
    source?.setData(rings(lat, lon));
    m.easeTo({ center: [lon, lat], duration: 400 });
  }, [lat, lon]);

  return (
    <div
      ref={holder}
      className={`h-[320px] w-full overflow-hidden rounded-[12px] border border-line sm:h-[420px] ${className}`}
      role="application"
      aria-label="Site location map. Tap to move the pin."
    />
  );
}
