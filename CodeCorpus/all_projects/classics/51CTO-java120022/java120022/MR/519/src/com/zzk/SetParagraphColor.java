package com.zzk;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import com.itextpdf.text.BaseColor;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.FontFactory;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;

public class SetParagraphColor {
	public static void main(String[] args){
		Document document = new Document();// 创建文档对象
		try {
			PdfWriter.getInstance(document, new FileOutputStream("c:\\Java类库参考手册.pdf"));// 关联文档对象与输出流
			document.open();// 打开文档
			Paragraph paragraph = new Paragraph(new Paragraph("JFrame Class Member List",FontFactory.getFont(FontFactory.HELVETICA, 30, BaseColor.BLUE )));// 创建段落定义字体并添加内容
			paragraph.setFirstLineIndent(100); // 设置段落首先缩进
			document.add(paragraph);// 向文档添加段落
			document.close();// 关闭文档
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		}
	}
}
