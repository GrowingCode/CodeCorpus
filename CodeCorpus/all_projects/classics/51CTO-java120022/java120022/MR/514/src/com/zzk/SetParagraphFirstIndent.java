package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;


public class SetParagraphFirstIndent {
	public static void main(String[] args){
		Document document = new Document();// 创建文档对象
		try {
			PdfWriter.getInstance(document, new FileOutputStream("c:\\公司简介.pdf"));// 关联文档对象与输出流
			document.open();// 打开文档
			Paragraph P1 = new Paragraph("MR company was builded in 1999!");// 创建段落并添加内容
			P1.setFirstLineIndent(20); 		//设置段落首先缩进
			document.add(new Paragraph(P1));//向文档添加段落
			Paragraph P2 = new Paragraph("Company own about for fifty employees.");//创建段落并添加内容
			document.add(P2);		//向文档添加段落
			document.close();		//关闭文档
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		}
	}
}
