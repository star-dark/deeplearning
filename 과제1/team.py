class Student :
    il = []
    nl = []
    def __init__(self,name,id) :
        self.name = name
        self.id = id
        Student.nl.append(self.name)
        Student.il.append(self.id)
    def show(self):
        tm = len(Student.nl)
        print("total member: ",tm)
        for i in range(tm):
            print(i," ",Student.nl[i]," ",Student.il[i])
        if tm<3:
            print(3-tm,"명의 추가할 팀원이 있습니다")
        else :
            print("완성된 팀구성입니다.")
a = Student("kim",60192516)
a.show()
b = Student("Cha",60192559)
b.show()
c = Student("Jo",60192557)
c.show()