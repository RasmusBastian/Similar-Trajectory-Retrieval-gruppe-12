pub const Coordinate = struct { lat: f32, lon: f32 };
const Timetype = enum { DEPARTURE, ARRIVAL };
const Time = struct {
    hour: u8,
    minute: u8,
    pub fn isValidTime(self: Time) bool {
        return self.hour < 24 and self.minute < 60;
    }
};
pub const JourneyInput = struct { start_coordinate: Coordinate, end_coordinate: Coordinate, timetype: Timetype, time: Time };

const LineString = struct {
    points: []Coordinate,

    pub fn init(points: []Coordinate) LineString {
        return LineString{ .points = points };
    }

    pub fn length(self: LineString) f32 {
        var sum: f32 = 0;
        for (self.points[0 .. self.points.len - 1], 0..) |p, i| {
            const next = self.points[i + 1];
            const dx = next.lat - p.lat;
            const dy = next.lon - p.lon;
            sum += @sqrt(dx * dx + dy * dy);
        }
        return sum;
    }
};

pub const GeoJsonLineString = struct {
    type: []const u8 = "LineString",
    coordinates: []const [2]f32,
};

pub const JourneyResponse = struct {
    linestring: GeoJsonLineString,
    traversalTime: f64,
    length: f64,
};
