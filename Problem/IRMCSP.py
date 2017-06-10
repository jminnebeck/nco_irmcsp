from random import random, randrange, choice
import numpy as np
import datetime
from copy import copy, deepcopy
import math


class IRMCSP:

    def __init__(self, instance=1):
        self.current_version_note = None

        self.instance = instance
        self.semester = None

        self.nr_saved_solutions = 0

        self.start_date = datetime.date(day=2, month=10, year=2017)
        self.end_date = None

        self.first_hour = 8
        self.last_hour = 20

        self.nr_weeks = None
        self.nr_days = 5
        self.nr_slots = (self.last_hour - self.first_hour) * 4
        self.nr_groups = 34

        self.nr_meetings = 0

        self.groups = {}
        self.series = {}
        self.courses = {}
        self.meetings = {}
        self.preferences = {}
        self.constraints = {}
        self.room_types = {}
        self.rooms = {}
        self.domains_as_categories = {}
        self.domains_as_calendar = {}

        self.int_to_indices = {}

        self.time_by_index = [[0, 0]]
        for hour in range(8, 20):
            for minute in range(0, 60, 15):
                self.time_by_index.append([hour, minute])

    def read_data(self, conn, solution):
        cursor = conn.cursor()
        sub_cursor = conn.cursor()

        sql = "SELECT * FROM instanzen WHERE instanzen.instanz_id = " + str(self.instance)
        cursor.execute(sql)
        row = cursor.fetchone()
        assert row is not None

        self.semester = row.get("semester")
        self.nr_weeks = row.get("wochen_überschreiben")

        sql = "SELECT * FROM semester WHERE semester.semester_id = " + str(self.semester)
        cursor.execute(sql)
        row = cursor.fetchone()
        assert row is not None

        if self.nr_weeks == 0:
            self.nr_weeks = row.get("wochen")
            # self.start_date = row.get("anfang")
            # self.end_date = row.get("ende")
        # else:
            # self.start_date = row.get("anfang")
            # self.end_date = self.start_date + datetime.timedelta(days=(7 * (self.nr_weeks - 1)) - 1)

        sql = "SELECT MAX(lösungs_id) AS max_id FROM lösungen"
        cursor.execute(sql)
        row = cursor.fetchone()
        if row is None:
            self.nr_saved_solutions = 0
        else:
            if row.get("max_id") is None:
                self.nr_saved_solutions = 0
            else:
                self.nr_saved_solutions = row.get("max_id")

        for g in range(1, self.nr_groups + 1):
            new_group = Group()
            new_group.id = g
            self.groups[g] = new_group

        sql = "SELECT * FROM reihen " \
              "WHERE reihen.instanz = " + str(self.instance) + " AND reihen.semester = " + str(self.semester) + " ORDER BY reihen.reihen_id"
        cursor.execute(sql)
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            new_series = Series()
            new_series.id = row.get("reihen_id")
            new_series.title = row.get("bezeichnung")
            new_series.semester = row.get("semester")
            self.series[new_series.id] = copy(new_series)

        sql = "SELECT * FROM raum_arten ORDER BY raum_arten.raum_arten_id"
        cursor.execute(sql)
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            new_room_type = Room_Type()
            new_room_type.id = row.get("raum_arten_id")
            new_room_type.title = row.get("bezeichnung")
            new_room_type.max_groups = row.get("max_gruppen")
            if new_room_type.max_groups == 0 or new_room_type.max_groups is None:
                new_room_type.max_groups = self.nr_groups
            new_room_type.equipment = row.get("ausstattung")
            self.room_types[new_room_type.id] = copy(new_room_type)

        sql = "SELECT * FROM räume WHERE räume.instanz = " + str(self.instance) + " OR räume.instanz is Null ORDER BY räume.raum_id"
        cursor.execute(sql)
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            new_room = Room()
            new_room.db_key = id = row.get("raum_id")
            new_room.type = self.room_types[row.get("art")]
            new_room.title = row.get("bezeichnung")
            new_room.max_groups = row.get("max_gruppen")
            if new_room.max_groups == 0 or new_room.max_groups is None:
                new_room.max_groups = new_room.type.max_groups
            new_room.equipment = row.get("ausstattung")
            if new_room.equipment is None:
                new_room.equipment = new_room.type.equipment
            self.rooms[new_room.id] = copy(new_room)
            new_room.type.rooms[new_room.id] = self.rooms[new_room.id]

        sql = "SELECT * FROM veranstaltungen INNER JOIN reihen ON reihen.reihen_id = veranstaltungen.reihe  " \
              "WHERE reihen.instanz = " + str(self.instance) + " AND reihen.semester = " + str(self.semester) + " ORDER BY veranstaltungen.veranstaltungs_id"
        cursor.execute(sql)
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            new_course = Course()
            new_course.id = row.get("veranstaltungs_id")
            new_course.series = self.series[row.get("reihe")]
            new_course.series.courses[new_course.id] = new_course
            new_course.title = row.get("bezeichnung")
            new_course.type = row.get("art")
            new_course.duration = row.get("dauer")
            new_course.max_groups = row.get("max_gruppen")
            if new_course.max_groups == 0:
                new_course.max_groups = self.nr_groups
            new_course.sessions = row.get("sitzungen")
            new_course.required_times = row.get("prf_zeiten")
            new_course.required_days = row.get("prf_tage")
            new_course.required_weeks = row.get("prf_wochen")
            new_course.requires_room = row.get("benötigt_raum")
            new_course.required_room_type = row.get("prf_raum_art")
            if new_course.requires_room:
                if new_course.required_room_type is not None:
                    new_course.required_room_type = self.room_types[int(new_course.required_room_type)]
            new_course.required_equipment = row.get("prf_ausstattung")
            if type(new_course.required_equipment) is str and len(new_course.required_equipment) > 0:
                new_course.required_equipment = str(new_course.required_equipment).split(";")
            self.courses[new_course.id] = copy(new_course)

        sql = "SELECT * FROM präferenzen ORDER BY präferenzen.präferenz_id"
        cursor.execute(sql)
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            new_preference = Preference()
            new_preference.id = row.get("präferenz_id")
            new_preference.title = row.get("bezeichnung")
            new_preference.value = row.get("wert")
            self.preferences[new_preference.id] = copy(new_preference)

        for course in sorted(self.courses.values(), key=lambda x: x.id):
            for session in range(1, course.sessions + 1):
                course_key = str(course.id) + "_" + str(session)
                session_groups = list(sorted(self.groups.keys(), reverse=True))
                nr_meetings = math.ceil(self.nr_groups / course.max_groups)
                for meeting in range(1, nr_meetings + 1):
                    new_meeting = Meeting()
                    new_meeting.title = course.title + "_s_" + str(session) + "_t_" + str(meeting)
                    new_meeting.series = course.series
                    new_meeting.course = course
                    new_meeting.session = session
                    new_meeting.course_key = course_key
                    new_meeting.duration = course.duration
                    new_meeting.duration_in_slots = math.ceil(course.duration / 15)
                    new_meeting.max_groups = course.max_groups
                    while (session_groups and
                               ((len(new_meeting.groups) < new_meeting.max_groups) or
                                    (len(session_groups) < (new_meeting.max_groups / 2)))):
                        new_meeting.groups.append(session_groups.pop())
                    self.meetings[new_meeting.id] = copy(new_meeting)
                    course.meetings[new_meeting.id] = self.meetings[new_meeting.id]
                    course.series.meetings[new_meeting.id] = self.meetings[new_meeting.id]

        sql = "SELECT * FROM restriktionen ORDER BY restriktionen.restriktions_id"
        cursor.execute(sql)
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            new_constraint = Constraint()
            new_constraint.id = row.get("restriktions_id")
            new_constraint.check_id()
            new_constraint.title = row.get("bezeichnung")
            new_constraint.type = row.get("art")
            sub_sql = "SELECT art FROM restriktions_arten WHERE restriktions_arten.restriktions_art_id = " + str(
                new_constraint.type)
            sub_cursor.execute(sub_sql)
            sub_row = sub_cursor.fetchone()
            new_constraint.type = sub_row.get("art")
            new_constraint.pref_value = self.preferences[row.get("präferenz")].value
            if new_constraint.pref_value == 0:
                new_constraint.is_hard = True
            else:
                new_constraint.is_hard = False
            new_constraint.level = row.get("ebene")
            new_constraint.owner = row.get("inhaber")
            if new_constraint.owner is not None:
                if new_constraint.owner in self.courses:
                    new_constraint.owner = self.courses[new_constraint.owner]
                    self.courses[new_constraint.owner.id].constraints[new_constraint.id] = new_constraint
                else:
                    continue
            sub_sql = "SELECT * FROM restriktions_zuordnung WHERE restriktions_zuordnung.restriktion = " + str(
                new_constraint.id)
            sub_cursor.execute(sub_sql)
            while True:
                sub_row = sub_cursor.fetchone()
                if sub_row is None:
                    break
                new_constraint.targets[sub_row.get("veranstaltung")] = self.courses[sub_row.get("veranstaltung")]
                self.courses[sub_row.get("veranstaltung")].constraints[new_constraint.id] = new_constraint
            self.constraints[new_constraint.id] = copy(new_constraint)

        solution.constraints = []
        for constraint in self.constraints.values():
            if constraint.owner:
                for meeting in constraint.owner.meetings.values():
                    meeting_constraint = deepcopy(constraint)
                    meeting_constraint.owner = meeting.id
                    meeting_constraint.targets = []
                    for target_course in constraint.targets.values():
                        if constraint.level == "Veranstaltung":
                            meeting_constraint.targets += list(target_course.meetings)
                        elif constraint.level == "Gruppe":
                            for target_meetings in target_course.meetings.values():
                                is_target = True
                                for target_group in target_meetings.groups:
                                    if target_group not in meeting.groups:
                                        is_target = False
                                        break
                                if is_target:
                                    meeting_constraint.targets.append(target_meetings.id)
                    solution.constraints.append(meeting_constraint)

        self.nr_meetings = solution.nr_meetings = len(self.meetings)

        i = 0
        for room in range(len(self.rooms)):
            for week in range(self.nr_weeks):
                for day in range(self.nr_days):
                    for slot in range(self.nr_slots):
                        self.int_to_indices[i] = (room, week, day, slot)
                        i += 1

        solution.irmcsp = self
        solution.nr_groups = self.nr_groups

        self.create_domains()

        for m in range(self.nr_meetings):
            solution.groups_per_meeting[m] = len(self.meetings[m].groups)
            solution.groups_by_meeting[m] = self.meetings[m].groups
            solution.durations[m] = self.meetings[m].duration
            solution.durations_in_slots[m] = self.meetings[m].duration_in_slots

        solution.domains_as_categories = deepcopy(self.domains_as_categories)
        solution.domains_as_calendar = deepcopy(self.domains_as_calendar)

        solution.meeting_state = {}
        solution.calendar_state = np.zeros((len(self.rooms), self.nr_weeks, self.nr_days, self.nr_slots))
        solution.calendar = np.zeros_like(solution.calendar_state).astype(np.int16)
        solution.int_to_indices = deepcopy(self.int_to_indices)

        sub_cursor.close()
        cursor.close()

    def create_domains(self):
        self.domains_as_categories = {}
        self.domains_as_calendar = np.zeros((self.nr_meetings, len(self.rooms), self.nr_weeks, self.nr_days, self.nr_slots))

        for meeting in self.meetings.values():
            possible_weeks = []
            possible_days = []
            possible_slots = []
            possible_rooms = []

            for weeks in range(1, self.nr_weeks + 1):
                if meeting.course.required_weeks is None or meeting.course.required_weeks == str(weeks):
                    possible_weeks.append(1)
                else:
                    possible_weeks.append(0)
            assert len(possible_weeks) == self.nr_weeks

            for day in range(1, self.nr_days + 1):
                if meeting.course.required_days[day - 1] == "1":
                    possible_days.append(1)
                else:
                    possible_days.append(0)
            assert len(possible_days) == self.nr_days

            duration_in_slots = math.ceil(meeting.duration / 15)
            for start_slot in range(self.nr_slots):
                is_allowed = True
                for slot in range(start_slot, start_slot + duration_in_slots):
                    if slot < self.nr_slots:
                        if meeting.course.required_times[math.floor(slot / 4)] != "1":
                            is_allowed = False
                    else:
                        is_allowed = False
                if is_allowed:
                    possible_slots.append(1)
                else:
                    possible_slots.append(0)
            assert len(possible_slots) == self.nr_slots

            if meeting.course.requires_room:
                if meeting.course.required_room_type in self.room_types.values():
                    acceptable_rooms = self.room_types[meeting.course.required_room_type.id].rooms
                else:
                    acceptable_rooms = self.rooms
                for room in self.rooms:
                    if room in acceptable_rooms:
                        if meeting.max_groups <= math.ceil(self.rooms[room].max_groups * 1.25):
                            if self.rooms[room].max_groups <= math.ceil(meeting.max_groups * 1.5):
                                if meeting.course.required_equipment is not None:
                                    has_equipment = True
                                    for equipment in meeting.course.required_equipment:
                                        if equipment not in room.equipment:
                                            has_equipment = False
                                            break
                                    if has_equipment:
                                        possible_rooms.append(1)
                                    else:
                                        possible_rooms.append(0)
                                else:
                                    possible_rooms.append(1)
                            else:
                                possible_rooms.append(0)
                        else:
                            possible_rooms.append(0)
                    else:
                        possible_rooms.append(0)
            else:
                for x in range(0, len(self.rooms)):
                    possible_rooms.append(1)
            assert len(possible_rooms) == len(self.rooms)

            dbg_count = 0
            for room in range(0, len(possible_rooms)):
                for week in range(0, len(possible_weeks)):
                    for day in range(0, len(possible_days)):
                        for time in range(0, len(possible_slots)):
                            if possible_rooms[room] == 1 and possible_weeks[week] == 1 and possible_days[day] == 1 and possible_slots[time] == 1:
                                self.domains_as_calendar[meeting.id, room, week, day, time] = 1
                                dbg_count += 1

            if dbg_count == 0:
                print("empty domain error")
            assert dbg_count > 0

            self.domains_as_categories[meeting.id] = {"rooms": np.array(possible_rooms),
                                                      "weeks": np.array(possible_weeks),
                                                      "days": np.array(possible_days),
                                                      "slots": np.array(possible_slots)}


    def write_solution(self, conn, solution):
        cursor = conn.cursor()

        sql = "select * from lösungen"
        cursor.execute(sql)

        timestamp = datetime.datetime.now()
        sql = "INSERT INTO lösungen(lösungs_id, instanz, bezeichnung, platziert, wert, bemerkung) VALUES " \
              "({}, {}, '{}', {}, {}, '{}')".format(solution.id,
                                                    self.instance,
                                                    "{}, {}, {}, {}".format(timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                                                            "{:.4f}".format(solution.placed_ratio),
                                                                            "{:.2f}".format(solution.pref_value),
                                                                            solution.actor_id),
                                                    "{:.4f}".format(solution.placed_ratio),
                                                    "{:.2f}".format(solution.pref_value),
                                                    self.current_version_note)

            #   + str(solution.id) + ", " + str(self.instance) + ", '" + timestamp.strftime("%Y-%m-%d %H:%M:%S") + " " + str(solution.actor_id) + "', " + str(
            # solution.placed_ratio) + ", " + str(solution.pref_value) + ", " + "'" + self.current_version_note + "')"

        cursor.execute(sql)
        cursor.commit()

        # Termindaten in Liste für executemany übertragen
        sql_list = []

        for meeting in self.meetings.values():
            if meeting.id in solution.placed_values:
                if meeting.groups:
                    for group in meeting.groups:
                        entry = []
                        entry.append(solution.id)
                        entry.append(self.instance)
                        entry.append(meeting.title)
                        entry.append(meeting.series.id)
                        entry.append(meeting.course.id)
                        entry.append(group)
                        entry.append(solution.placed_values[meeting.id].instructor)
                        entry.append(self.rooms[int(solution.placed_values[meeting.id].room) + 1].db_key)
                        entry.append(int(solution.placed_values[meeting.id].week) + 1)
                        meeting_date = self.start_date + datetime.timedelta(weeks=int(solution.placed_values[meeting.id].week),
                                                                            days=int(solution.placed_values[meeting.id].day))
                        entry.append(meeting_date)
                        start_hour, start_minute = self.time_by_index[solution.placed_values[meeting.id].start]
                        end_hour, end_minute = self.time_by_index[solution.placed_values[meeting.id].end]
                        entry.append(datetime.time(start_hour, start_minute))
                        entry.append(datetime.time(end_hour, end_minute))
                        entry.append(solution.placed_values[meeting.id].pref_value)
                        sql_list.append(entry)
                else:
                    entry = []
                    entry.append(solution.id)
                    entry.append(self.instance)
                    entry.append(meeting.title)
                    entry.append(meeting.series.id)
                    entry.append(meeting.course.id)
                    entry.append(None)
                    entry.append(solution.placed_values[meeting.id].instructor)
                    entry.append(self.rooms[int(solution.placed_values[meeting.id].room)].db_key)
                    entry.append(int(solution.placed_values[meeting.id].week) + 1)
                    meeting_date = self.start_date + datetime.timedelta(weeks=solution.placed_values[meeting.id].week,
                                                                              days=solution.placed_values[meeting.id].day)
                    entry.append(meeting_date)
                    start_hour, start_minute = self.time_by_index[solution.placed_values[meeting.id].start]
                    end_hour, end_minute = self.time_by_index[solution.placed_values[meeting.id].end]
                    entry.append(datetime.time(start_hour, start_minute))
                    entry.append(datetime.time(end_hour, end_minute))
                    entry.append(solution.placed_values[meeting.id].pref_value)
                    sql_list.append(entry)
            else:
                entry = []
                entry.append(solution.id)
                entry.append(self.instance)
                entry.append(meeting.title)
                entry.append(meeting.series.id)
                entry.append(meeting.course.id)
                entry.append(None)
                entry.append(None)
                entry.append(None)
                entry.append(None)
                entry.append(None)
                entry.append(None)
                entry.append(None)
                entry.append(None)
                sql_list.append(entry)

        sql = "INSERT INTO termine (lösung, instanz, bezeichnung, reihe, veranstaltung, gruppe, dozent, raum, woche, datum, von, " \
              "bis, wert) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "

        cursor.executemany(sql, sql_list)
        cursor.commit()
        cursor.close()

class Value:

    def __init__(self, meeting, indices, duration_in_slots):
        self.id = 0
        self.meeting = meeting
        self.indices = indices
        self.instructor = None
        self.room = indices[0]
        self.week = self.indices[1]
        self.day = self.indices[2]
        self.start = self.indices[3]
        self.end =  self.start + duration_in_slots - 1
        self.pref_value = 0

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "Meeting {}, {}".format(self.meeting, self.indices)

class Constraint:

    next_uid = 1

    def __init__(self):
        self.id = 0
        self.title = ""
        self.type = ""
        self.pref_value = 0
        self.is_hard = True
        self.level = ""
        self.owner = None
        self.targets = {}

    def check_id(self):
        if self.id == 0 or self.id is None:
            self.id = Constraint.next_uid
            Constraint.next_uid += 1

    def check_consistency(self, solution, meeting, value):
        if meeting == self.owner or meeting in self.targets:
            if self.type == "Präzedenz":
                return self.evaluate_precedence(solution, meeting, value)
            # elif self.type == "Abfolge":
            #     return self.evaluate_order(solution, value)
            # elif self.type == "Konsekutive_Gruppen":
            #     return self.evaluate_group_consecutiveness(solution, value)
            # elif self.type == "Spätes_Ende":
            #     return self.evaluate_late_end(solution, value)

    def evaluate_precedence(self, solution, meeting, value):
        conflicts = []

        if self.level == "Veranstaltung":
            if meeting == self.owner:
                owner_value = value
                owner_end = (owner_value.week, owner_value.day, owner_value.end)
                for target in self.targets:
                    if target in solution.placed_values:
                        target_value = solution.placed_values[target]
                        target_start = (target_value.week, target_value.day, target_value.start)
                        if owner_end >= target_start:
                            if self.is_hard:
                                conflicts.append(target)
                            else:
                                owner_value.pref_value *= (1 - self.pref_value)
                                target_value.pref_value *= (1 - self.pref_value)

            elif meeting in self.targets:
                if self.owner in solution.placed_values:
                    owner_value = solution.placed_values[self.owner]
                    owner_end = (owner_value.week, owner_value.day, owner_value.end)
                    target_value = value
                    target_start = (target_value.week, target_value.day, target_value.start)
                    if owner_end >= target_start:
                        if self.is_hard:
                            conflicts.append(self.owner)
                        else:
                            owner_value.pref_value *= (1 - self.pref_value)
                            target_value.pref_value *= (1 - self.pref_value)

        elif self.level == "Gruppe":
            if meeting == self.owner:
                for group in solution.groups_by_meeting[meeting]:
                    owner_value = value
                    owner_end = (owner_value.week, owner_value.day, owner_value.end)
                    for target in self.targets:
                        if target in solution.placed_values:
                            if group in solution.groups_by_meeting[target]:
                                target_value = solution.placed_values[target]
                                target_start = (target_value.week, target_value.day, target_value.start)
                                if owner_end >= target_start:
                                    if self.is_hard:
                                        if target not in conflicts:
                                            conflicts.append(target)
                                    else:
                                        owner_value.pref_value *= (1 - self.pref_value / solution.groups_per_meeting[meeting])
                                        target_value.pref_value *= (1 - self.pref_value / solution.groups_per_meeting[target])


            elif meeting in self.targets:
                if self.owner in solution.placed_values:
                    for group in solution.groups_by_meeting[meeting]:
                        owner_value = solution.placed_values[self.owner]
                        owner_end = (owner_value.week, owner_value.day, owner_value.end)
                        if group in solution.groups_by_meeting[meeting]:
                            target_value = value
                            target_start = (target_value.week, target_value.day, target_value.start)
                            if owner_end >= target_start:
                                if self.is_hard:
                                    if self.owner not in conflicts:
                                        conflicts.append(self.owner)
                                else:
                                    owner_value.pref_value *= (1 - self.pref_value / solution.groups_per_meeting[self.owner])
                                    target_value.pref_value *= (1 - self.pref_value / solution.groups_per_meeting[meeting])

        return conflicts

    # def evaluate_order(self, solution, value=None):
    #     conflict_value = 0
    #     conflicts = []
    #
    #     return conflicts, conflict_value
    #
    # def evaluate_group_consecutiveness(self, solution, value=None):
    #     conflict_value = 0
    #     conflicts = []
    #
    #     # if value is not None:
    #     #     previous_group = 0
    #     #     if value.groups:
    #     #         for group in sorted(value.groups):
    #     #             if previous_group != 0:
    #     #                 if group - previous_group > 1:
    #     #                     conflict_value += self.pref_value
    #     #             previous_group = group
    #     # else:
    #     #     for meeting in solution.placed_meetings.values():
    #     #         previous_group = 0
    #     #         if meeting.assigned_value.groups:
    #     #             for group in sorted(meeting.assigned_value.groups):
    #     #                 if previous_group != 0:
    #     #                     if group - previous_group > 1:
    #     #                         conflict_value += self.pref_value
    #     #                 previous_group = group
    #
    #     return conflicts, conflict_value
    #
    # def evaluate_late_end(self, solution, value=None):
    #     conflict_value = 0
    #     conflicts = []
    #
    #     if value is not None:
    #         if value.end > datetime.time(hour=16):
    #             if self.is_hard:
    #                 conflicts.append(value)
    #             else:
    #                 tst = value.end - datetime.time(hour=16)
    #                 conflict_value += self.pref_value * len(value.meeting.groups)
    #     else:
    #         for value in solution.placed_values.values():
    #             if value.end > datetime.time(hour=16):
    #                 if self.is_hard:
    #                     conflicts.append(value)
    #                 else:
    #                     tst = value.end - datetime.time(hour=16)
    #                     conflict_value += self.pref_value
    #
    #     return conflicts, conflict_value

class Meeting:

    next_uid = 0

    def __init__(self):
        self.id = Meeting.next_uid
        Meeting.next_uid += 1

        self.assigned_value = None
        self.title = ""
        self.series = None
        self.course = None
        self.session = 1
        self.course_key = ""
        self.max_groups = 0
        self.groups = []
        self.duration = 0
        self.duration_in_slots = 0
        self.constraints = {}
        self.pref_value = 0

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "Meeting {}, {}".format(self.id, self.title)

class Course:

    def __init__(self):
        self.id = 0
        self.series = None
        self.title = ""
        self.type = ""
        self.duration = 0
        self.max_groups = 0
        self.sessions = 1
        self.required_weeks = ""
        self.required_days = ""
        self.required_times = ""
        self.requires_room = True
        self.required_room_type = ""
        self.required_equipment = ""
        self.constraints = {}
        self.meetings = {}

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "Course {}, {}".format(self.id, self.title)

class Series:

    def __init__(self):
        self.id = 0
        self.title = ""
        self.semester = ""
        self.courses = {}
        self.meetings = {}

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "Series {}, {}".format(self.id, self.title)

class Group:

    def __init__(self):
        self.id = 0
        self.size = 10
        self.students = {}


class Room:

    next_uid = 1

    def __init__(self):
        self.id = Room.next_uid
        Room.next_uid += 1

        self.db_key = 0
        self.title = ""
        self.type = None
        self.max_groups = 0
        self.equipment = []
        self.pref_value = 0

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "Room {}, {}, Size {} ({})".format(self.id, self.title, self.max_groups, self.type.title)

class Room_Type:

    def __init__(self):
        self.id = 0
        self.title = ""
        self.max_groups = 0
        self.equipment = []
        self.rooms = {}

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "Room_type {}, {}, Size {}".format(self.id, self.title, self.max_groups)

class Preference:

    def __init__(self):
        self.id = 0
        self.title = ""
        self.value = 0